"""Tests for the `Wannier90WorkChain` class."""

import io
from pathlib import Path
from types import SimpleNamespace

from plumpy.process_states import ProcessState

from aiida import orm
from aiida.common import AttributeDict, LinkType

from aiida_quantumespresso.calculations.helpers import pw_input_helper

from aiida_wannier90_workflows.utils.cwf import cwf_window_function


def generate_cwf_file_contents():
    """Return synthetic ``eig`` and ``amn`` contents for CWF fitting tests."""
    import numpy as np

    energies = np.linspace(-5.0, 5.0, 21)
    projectability = cwf_window_function(
        energies,
        mu_min=-3.5,
        mu_max=2.0,
        sigma_min=0.5,
        sigma_max=0.25,
    )
    projectability = np.clip(projectability, 0.0, 1.0)

    eig_lines = [f"{band_index} 1 {energy:.16f}" for band_index, energy in enumerate(energies, start=1)]
    amn_lines = ["generated for test", f"{len(energies)} 1 1"]
    for band_index, value in enumerate(projectability, start=1):
        amplitude = float(np.sqrt(value))
        amn_lines.append(f"1 {band_index} 1 {amplitude:.16f} 0.0")

    return "\n".join(eig_lines), "\n".join(amn_lines)


def test_scdm(
    generate_workchain_wannier90,
    fixture_localhost,
    generate_remote_data,
    generate_bands_data,
    generate_projection_data,
    generate_calc_job_node,
):  # pylint: disable=redefined-outer-name,too-many-statements
    """Test instantiating the WorkChain, then mock its process, by calling methods in the ``spec.outline``."""

    workchain = generate_workchain_wannier90()
    assert workchain.setup() is None

    # run scf
    scf_workchain = workchain.run_scf()["workchain_scf"]

    # mock scf outputs
    remote = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    remote.store()
    remote.base.links.add_incoming(scf_workchain, link_type=LinkType.RETURN, link_label="remote_folder")

    params = orm.Dict({"fermi_energy": 6.0, "number_of_electrons": 8})
    params.store()
    params.base.links.add_incoming(scf_workchain, link_type=LinkType.RETURN, link_label="output_parameters")

    scf_workchain.set_process_state(ProcessState.FINISHED)
    scf_workchain.set_exit_status(0)
    workchain.ctx.workchain_scf = scf_workchain

    pw_input_helper(scf_workchain.inputs.pw.parameters.get_dict(), scf_workchain.inputs.pw.structure)
    assert workchain.inspect_scf() is None
    assert workchain.ctx.current_folder == remote

    # run nscf
    nscf_workchain = workchain.run_nscf()["workchain_nscf"]

    # mock nscf outputs
    remote = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    remote.store()
    remote.base.links.add_incoming(nscf_workchain, link_type=LinkType.RETURN, link_label="remote_folder")

    nscf_workchain.set_process_state(ProcessState.FINISHED)
    nscf_workchain.set_exit_status(0)
    workchain.ctx.workchain_nscf = nscf_workchain

    pw_input_helper(
        nscf_workchain.inputs.pw.parameters.get_dict(),
        nscf_workchain.inputs.pw.structure,
    )
    assert workchain.ctx.workchain_nscf.inputs.pw.parent_folder == workchain.ctx.workchain_scf.outputs.remote_folder
    assert workchain.inspect_nscf() is None
    assert workchain.ctx.current_folder == remote

    # mock run projwfc
    projwfc_workchain = workchain.run_projwfc()["workchain_projwfc"]

    # mock projwfc outputs
    bands_data = generate_bands_data()
    bands_data.store()
    bands_data.base.links.add_incoming(projwfc_workchain, link_type=LinkType.RETURN, link_label="bands")

    # Set 8 orbitals for workchain.sanity_check()
    projection_data = generate_projection_data(8)
    projection_data.store()
    projection_data.base.links.add_incoming(projwfc_workchain, link_type=LinkType.RETURN, link_label="projections")

    projwfc_workchain.set_process_state(ProcessState.FINISHED)
    projwfc_workchain.set_exit_status(0)
    workchain.ctx.workchain_projwfc = projwfc_workchain

    assert (
        workchain.ctx.workchain_projwfc.inputs.projwfc.parent_folder
        == workchain.ctx.workchain_nscf.outputs.remote_folder
    )
    assert workchain.inspect_projwfc() is None

    # mock run wannier90 pp
    w90pp_workchain = workchain.run_wannier90_pp()["workchain_wannier90_pp"]

    # The wannier90 step will use `get_last_calcjob` to retrieve input parameters of the calcjob
    entry_point_calc_job = "wannier90.wannier90"
    calcjob = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        inputs={"parameters": orm.Dict()},
        store=False,
    )
    calcjob.set_process_state(ProcessState.FINISHED)
    calcjob.set_exit_status(0)
    calcjob.base.links.add_incoming(
        workchain.inputs.structure,
        link_type=LinkType.INPUT_CALC,
        link_label="structure",
    )
    calcjob.base.links.add_incoming(w90pp_workchain, link_type=LinkType.CALL_CALC, link_label="iteration_01")
    calcjob.store()

    assert w90pp_workchain.called_descendants == [calcjob]

    # mock wannier90 outputs
    nnkp_file = orm.SinglefileData(io.BytesIO(b"content"))
    nnkp_file.store()
    nnkp_file.base.links.add_incoming(w90pp_workchain, link_type=LinkType.RETURN, link_label="nnkp_file")

    w90pp_workchain.set_process_state(ProcessState.FINISHED)
    w90pp_workchain.set_exit_status(0)
    workchain.ctx.workchain_wannier90_pp = w90pp_workchain

    assert workchain.inspect_wannier90_pp() is None

    # mock run pw2wannier90
    pw2wan_workchain = workchain.run_pw2wannier90()["workchain_pw2wannier90"]

    # mock pw2wannier90 outputs
    remote = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    remote.store()
    remote.base.links.add_incoming(pw2wan_workchain, link_type=LinkType.RETURN, link_label="remote_folder")

    pw2wan_workchain.set_process_state(ProcessState.FINISHED)
    pw2wan_workchain.set_exit_status(0)
    workchain.ctx.workchain_pw2wannier90 = pw2wan_workchain

    assert (
        workchain.ctx.workchain_pw2wannier90.inputs.pw2wannier90.parent_folder
        == workchain.ctx.workchain_nscf.outputs.remote_folder
    )
    assert workchain.inspect_pw2wannier90() is None
    assert workchain.ctx.current_folder == remote

    # mock run wannier90
    w90_workchain = workchain.run_wannier90()["workchain_wannier90"]

    # mock wannier90 outputs
    remote = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    remote.store()
    remote.base.links.add_incoming(w90_workchain, link_type=LinkType.RETURN, link_label="remote_folder")

    w90_workchain.set_process_state(ProcessState.FINISHED)
    w90_workchain.set_exit_status(0)
    workchain.ctx.workchain_wannier90 = w90_workchain

    assert workchain.inspect_wannier90() is None
    assert workchain.ctx.current_folder == remote

    assert workchain.results() is None

    assert all(_ in workchain.outputs for _ in ("scf", "nscf", "projwfc", "wannier90_pp", "pw2wannier90", "wannier90"))


def test_prepare_pw2wannier90_inputs_auto_cwf(
    generate_workchain,
    generate_inputs_wannier90,
    generate_remote_data,
    fixture_localhost,
):
    """Test that CWF fitting configures pw2wannier90 correctly."""
    inputs = generate_inputs_wannier90()
    inputs["auto_cwf_parameters"] = orm.Bool(True)
    inputs["pw2wannier90"]["pw2wannier90"]["parameters"] = orm.Dict({"inputpp": {}})
    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)

    workchain.ctx.current_folder = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    nnkp_file = orm.SinglefileData(io.BytesIO(b"content")).store()
    workchain.ctx.workchain_wannier90_pp = SimpleNamespace(outputs=AttributeDict({"nnkp_file": nnkp_file}))

    prepared = workchain.prepare_pw2wannier90_inputs()
    parameters = prepared["pw2wannier90"]["parameters"].get_dict()["inputpp"]
    settings = prepared["pw2wannier90"]["settings"].get_dict()

    assert parameters["atom_proj"] is True
    assert "aiida.amn" in settings["additional_retrieve_list"]
    assert "aiida.eig" in settings["additional_retrieve_list"]


def test_fit_cwf_parameters(
    tmp_path,
    monkeypatch,
    generate_workchain,
    generate_inputs_wannier90,
):
    """Test fitting CWF parameters from retrieved pw2wannier90 files."""
    inputs = generate_inputs_wannier90()
    inputs["auto_cwf_parameters"] = orm.Bool(True)
    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)

    eig_content, amn_content = generate_cwf_file_contents()
    folder = tmp_path / "retrieved"
    folder.mkdir()
    (folder / "aiida.eig").write_text(eig_content, encoding="utf-8")
    (folder / "aiida.amn").write_text(amn_content, encoding="utf-8")

    retrieved = orm.FolderData()
    retrieved.put_object_from_tree(Path(folder))
    retrieved.store()

    last_calc = SimpleNamespace(outputs=AttributeDict({"retrieved": retrieved}))
    monkeypatch.setattr(
        "aiida_wannier90_workflows.utils.workflows.get_last_calcjob",
        lambda _: last_calc,
    )

    workchain.ctx.workchain_pw2wannier90 = SimpleNamespace()

    assert workchain.fit_cwf_parameters() is None
    parameters = workchain.ctx.cwf_parameters.get_dict()
    assert set(parameters) == {
        "cwf_mu_max",
        "cwf_mu_min",
        "cwf_sigma_max",
        "cwf_sigma_min",
    }
    assert parameters["cwf_sigma_max"] > 0.0
    assert parameters["cwf_sigma_min"] >= 0.0
    dimensions = workchain.ctx.cwf_dimensions.get_dict()
    assert dimensions["num_bands"] == 21
    assert dimensions["num_wann"] == 1


def test_prepare_wannier90_pp_inputs_auto_cwf(
    generate_workchain,
    generate_inputs_wannier90,
):
    """Test that Wannier90 postproc inputs are sanitized for CWF runs."""
    inputs = generate_inputs_wannier90()
    inputs["auto_cwf_parameters"] = orm.Bool(True)
    inputs["wannier90"]["wannier90"]["parameters"] = orm.Dict(
        {
            "fermi_energy": 0.0,
            "dis_proj_min": 0.01,
            "dis_proj_max": 0.95,
            "dis_froz_min": -5.0,
            "dis_froz_max": 5.0,
            "dis_win_min": -10.0,
            "dis_win_max": 10.0,
        }
    )
    inputs["wannier90"]["wannier90"]["projections"] = orm.List(list=["Si:s"])
    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)
    workchain.ctx.current_structure = workchain.inputs.structure

    prepared = workchain.prepare_wannier90_pp_inputs()
    parameters = prepared["wannier90"]["parameters"].get_dict()

    assert parameters["auto_projections"] is True
    assert parameters["fermi_energy"] == 0.0
    assert "dis_proj_min" not in parameters
    assert "dis_proj_max" not in parameters
    assert "dis_froz_min" not in parameters
    assert "dis_froz_max" not in parameters
    assert "dis_win_min" not in parameters
    assert "dis_win_max" not in parameters
    assert "projections" not in prepared["wannier90"]


def test_prepare_wannier90_inputs_auto_cwf(
    monkeypatch,
    generate_workchain,
    generate_inputs_wannier90,
    generate_remote_data,
    fixture_localhost,
):
    """Test that the final Wannier90 inputs include the CWF settings."""
    inputs = generate_inputs_wannier90()
    inputs["auto_cwf_parameters"] = orm.Bool(True)
    inputs["wannier90"]["wannier90"]["parameters"] = orm.Dict({"cwf_sigma_min": -1000})
    inputs["wannier90"]["wannier90"]["projections"] = orm.List(list=["Si:s"])
    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)

    workchain.ctx.current_folder = generate_remote_data(computer=fixture_localhost, remote_path="/path/on/remote")
    workchain.ctx.cwf_parameters = orm.Dict(
        {
            "cwf_mu_max": 1.23,
            "cwf_mu_min": 0.45,
            "cwf_sigma_max": 0.67,
            "cwf_sigma_min": 0.89,
        }
    )
    workchain.ctx.cwf_dimensions = orm.Dict(
        {
            "num_bands": 24,
            "num_wann": 8,
        }
    )

    last_calc = SimpleNamespace(
        inputs=AttributeDict(
            {
                "parameters": orm.Dict(
                    {
                        "cwf_sigma_min": -1000,
                        "dis_proj_min": 0.01,
                        "dis_proj_max": 0.95,
                        "dis_froz_min": -5.0,
                        "dis_froz_max": 5.0,
                        "dis_win_min": -10.0,
                        "dis_win_max": 10.0,
                    }
                ),
                "metadata": {"options": {}},
                "settings": orm.Dict({}),
                "projections": orm.List(list=["Si:s"]),
            }
        )
    )
    monkeypatch.setattr(
        "aiida_wannier90_workflows.utils.workflows.get_last_calcjob",
        lambda _: last_calc,
    )
    workchain.ctx.workchain_wannier90_pp = SimpleNamespace()

    prepared = workchain.prepare_wannier90_inputs()
    parameters = prepared["wannier90"]["parameters"].get_dict()

    assert parameters["auto_projections"] is True
    assert parameters["guiding_centres"] is True
    assert parameters["num_iter"] == 0
    assert parameters["dis_num_iter"] == 0
    assert parameters["use_cwf_method"] is True
    assert parameters["cwf_delta"] == 1e-12
    assert parameters["cwf_mu_max"] == 1.23
    assert parameters["cwf_mu_min"] == 0.45
    assert parameters["cwf_sigma_max"] == 0.67
    assert parameters["cwf_sigma_min"] == -1000
    assert parameters["num_bands"] == 24
    assert parameters["num_wann"] == 8
    assert "dis_proj_min" not in parameters
    assert "dis_proj_max" not in parameters
    assert "dis_froz_min" not in parameters
    assert "dis_froz_max" not in parameters
    assert "dis_win_min" not in parameters
    assert "dis_win_max" not in parameters
    assert "projections" not in prepared["wannier90"]


def test_plot_cwf_fit(
    tmp_path,
    monkeypatch,
):
    """Test plotting the Closest Wannier fitting from a workchain."""
    import matplotlib.pyplot as plt

    from aiida_wannier90_workflows.utils.workflows.plot.bands import plot_cwf_fit
    from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

    eig_content, amn_content = generate_cwf_file_contents()
    folder = tmp_path / "retrieved"
    folder.mkdir()
    (folder / "aiida.eig").write_text(eig_content, encoding="utf-8")
    (folder / "aiida.amn").write_text(amn_content, encoding="utf-8")

    retrieved = orm.FolderData()
    retrieved.put_object_from_tree(Path(folder))
    retrieved.store()

    p2w_calc = SimpleNamespace(outputs=AttributeDict({"retrieved": retrieved}))
    w90_calc = SimpleNamespace(
        inputs=AttributeDict(
            {
                "parameters": orm.Dict(
                    {
                        "fermi_energy": 0.0,
                        "cwf_delta": 1e-12,
                    }
                )
            }
        )
    )

    p2w_workchain = SimpleNamespace()
    w90_workchain = SimpleNamespace()

    class _Outgoing:
        def __init__(self, node):
            self.node = node

        def one(self):
            return SimpleNamespace(node=self.node)

    workchain = SimpleNamespace(
        process_class=Wannier90WorkChain,
        process_label="Wannier90WorkChain",
        pk=999,
        inputs=AttributeDict(
            {
                "structure": SimpleNamespace(get_formula=lambda: "Si2"),
                "cwf_sigma_factor": orm.Float(3.0),
            }
        ),
        base=SimpleNamespace(
            links=SimpleNamespace(
                get_outgoing=lambda link_label_filter=None: _Outgoing(
                    p2w_workchain if link_label_filter == "pw2wannier90" else w90_workchain
                )
            )
        ),
    )

    def _mock_get_last_calcjob(node):
        if node is p2w_workchain:
            return p2w_calc
        if node is w90_workchain:
            return w90_calc
        raise ValueError(f"Unexpected node: {node}")

    monkeypatch.setattr(
        "aiida_wannier90_workflows.utils.workflows.get_last_calcjob",
        _mock_get_last_calcjob,
    )

    called = {"show": 0}

    def _show():
        called["show"] += 1

    monkeypatch.setattr(plt, "show", _show)

    plot_cwf_fit(workchain, save=False)

    assert called["show"] == 1
    plt.close("all")
