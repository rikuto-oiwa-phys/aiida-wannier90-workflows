"""Base class for Wannierisation workflow."""

# pylint: disable=protected-access
import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ExitCode
from aiida.engine.processes import ProcessBuilder, ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
    recursive_merge,
)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)

from .base.projwfc import ProjwfcBaseWorkChain
from .base.pw2wannier90 import Pw2wannier90BaseWorkChain
from .base.wannier90 import Wannier90BaseWorkChain

__all__ = ["validate_inputs", "Wannier90WorkChain"]


def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument,inconsistent-return-statements
    """Validate the inputs of the entire input namespace of `Wannier90WorkChain`."""
    # If no scf inputs are provided, the nscf inputs must include a parent folder.
    if "scf" not in inputs:
        if "nscf" in inputs and "parent_folder" not in inputs["nscf"]["pw"]:
            return "If skipping scf step, nscf inputs must have a `parent_folder`"

    # Cannot specify both `auto_energy_windows` and `scdm_proj`.
    pw2wannier_parameters = inputs["pw2wannier90"]["pw2wannier90"]["parameters"].get_dict()
    auto_energy_windows = inputs["wannier90"].get("auto_energy_windows", False)
    scdm_proj = pw2wannier_parameters["inputpp"].get("scdm_proj", False)
    if auto_energy_windows and scdm_proj:
        return "`auto_energy_windows` is incompatible with SCDM"

    # Cannot specify both `auto_energy_windows` and `shift_energy_windows`.
    shift_energy_windows = inputs["wannier90"].get("shift_energy_windows", False)
    if auto_energy_windows and shift_energy_windows:
        return "`auto_energy_windows` and `shift_energy_windows` are incompatible"


# pylint: disable=fixme,too-many-lines
class Wannier90WorkChain(ProtocolMixin, WorkChain):  # pylint: disable=too-many-public-methods
    """Workchain to obtain maximally localised Wannier functions (MLWF).

    Run the following steps:
        scf -> nscf -> projwfc -> wannier90 postproc -> pw2wannier90 -> wannier90
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        from .base.pw2wannier90 import (
            validate_inputs_base as validate_inputs_base_pw2wannier90,
        )
        from .base.wannier90 import (
            validate_inputs_base as validate_inputs_base_wannier90,
        )

        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData, help="The input structure.")
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help=("If True, work directories of all called calculation will be cleaned " "at the end of execution."),
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `PwBaseWorkChain` for the SCF calculation.",
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="nscf",
            exclude=("clean_workdir", "pw.structure", "pw.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `PwBaseWorkChain` for the NSCF calculation.",
            },
        )
        spec.expose_inputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            exclude=("clean_workdir", "projwfc.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `ProjwfcBaseWorkChain`.",
            },
        )
        spec.expose_inputs(
            Pw2wannier90BaseWorkChain,
            namespace="pw2wannier90",
            exclude=(
                "clean_workdir",
                "pw2wannier90.parent_folder",
                "pw2wannier90.nnkp_file",
            ),
            namespace_options={"help": "Inputs for the `Pw2wannier90BaseWorkChain`."},
        )
        spec.inputs["pw2wannier90"].validator = validate_inputs_base_pw2wannier90
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="wannier90",
            exclude=("clean_workdir", "wannier90.structure"),
            namespace_options={"help": "Inputs for the `Wannier90BaseWorkChain`."},
        )
        spec.inputs["wannier90"].validator = validate_inputs_base_wannier90

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_nscf)(
                cls.run_nscf,
                cls.inspect_nscf,
            ),
            if_(cls.should_run_projwfc)(
                cls.run_projwfc,
                cls.inspect_projwfc,
            ),
            cls.run_wannier90_pp,
            cls.inspect_wannier90_pp,
            cls.run_pw2wannier90,
            cls.inspect_pw2wannier90,
            if_(cls.should_fit_cwf_parameters)(
                cls.fit_cwf_parameters,
            ),
            cls.run_wannier90,
            cls.inspect_wannier90,
            cls.results,
        )

        spec.expose_outputs(PwBaseWorkChain, namespace="scf", namespace_options={"required": False})
        spec.expose_outputs(PwBaseWorkChain, namespace="nscf", namespace_options={"required": False})
        spec.expose_outputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            namespace_options={"required": False},
        )
        spec.expose_outputs(Pw2wannier90BaseWorkChain, namespace="pw2wannier90")
        spec.expose_outputs(Wannier90BaseWorkChain, namespace="wannier90_pp")
        spec.expose_outputs(Wannier90BaseWorkChain, namespace="wannier90")
        spec.output(
            "cwf_parameters",
            valid_type=orm.Dict,
            required=False,
            help="Automatically fitted Closest Wannier parameters.",
        )

        spec.input(
            "auto_cwf_parameters",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help=(
                "If True, fit CWF parameters from retrieved `aiida.eig` and "
                "`aiida.amn` before the final Wannier90 run."
            ),
        )
        spec.input(
            "cwf_sigma_factor",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            default=lambda: orm.Float(3.0),
            help="Shift factor applied to the fitted `mu_max`.",
        )
        spec.input(
            "cwf_delta",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            default=lambda: orm.Float(1e-12),
            help="Value written to the Wannier90 input as `cwf_delta`.",
        )

        spec.exit_code(
            420,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="the scf PwBaseWorkChain sub process failed",
        )
        spec.exit_code(
            430,
            "ERROR_SUB_PROCESS_FAILED_NSCF",
            message="the nscf PwBaseWorkChain sub process failed",
        )
        spec.exit_code(
            440,
            "ERROR_SUB_PROCESS_FAILED_PROJWFC",
            message="the ProjwfcBaseWorkChain sub process failed",
        )
        spec.exit_code(
            450,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90PP",
            message="the postproc Wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(
            460,
            "ERROR_SUB_PROCESS_FAILED_PW2WANNIER90",
            message="the Pw2wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(
            465,
            "ERROR_CWF_FILES_MISSING",
            message=("the retrieved `aiida.eig` or `aiida.amn` file required for " "CWF fitting is missing"),
        )
        spec.exit_code(
            466,
            "ERROR_CWF_FITTING_FAILED",
            message="fitting the Closest Wannier parameters failed",
        )
        spec.exit_code(
            470,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90",
            message="the Wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(480, "ERROR_SANITY_CHECK_FAILED", message="outputs sanity check failed")

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "wannier90.yaml"

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for various input arguments of the ``get_builder_from_protocol()`` method."""
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / "overrides" / "wannier90.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=unused-argument
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = None,
        electronic_type: ElectronicType = ElectronicType.METAL,
        spin_type: SpinType = SpinType.NONE,
        initial_magnetic_moments: dict = None,
        projection_type: WannierProjectionType = WannierProjectionType.SCDM,
        disentanglement_type: WannierDisentanglementType = None,
        frozen_type: WannierFrozenType = None,
        exclude_semicore: bool = True,
        external_projectors_path: str = None,
        plot_wannier_functions: bool = False,
        retrieve_hamiltonian: bool = False,
        retrieve_matrices: bool = False,
        print_summary: bool = True,
        summary: dict = None,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        The builder can be submitted directly by `aiida.engine.submit(builder)`.

        :param codes: a dictionary of ``Code`` instance for pw.x, pw2wannier90.x, wannier90.x, (optionally) projwfc.x.
        :type codes: dict
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param protocol: protocol to use, if not specified, the default will be used.
        :type protocol: str
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param electronic_type: indicate the electronic character of the system through ``ElectronicType`` instance.
        :param spin_type: indicate the spin polarization type to use through a ``SpinType`` instance.
        :param initial_magnetic_moments: optional dictionary that maps the initial magnetic moment of
        each kind to a desired value for a spin polarized calculation.
        Note that for ``spin_type == SpinType.COLLINEAR`` an initial guess for the magnetic moment
        is automatically set in case this argument is not provided.
        :param projection_type: indicate the Wannier initial projection type of the system
        through ``WannierProjectionType`` instance.
        Default to SCDM.
        :param disentanglement_type: indicate the Wannier disentanglement type of the system through
        ``WannierDisentanglementType`` instance. Default to None, which will choose the best type
        based on `projection_type`:
            For WannierProjectionType.SCDM, use WannierDisentanglementType.NONE
            For other WannierProjectionType, use WannierDisentanglementType.SMV
        :param frozen_type: indicate the Wannier disentanglement type of the system
        through ``WannierFrozenType`` instance. Default to None, which will choose
        the best frozen type based on `electronic_type` and `projection_type`.
            for ElectronicType.INSULATOR, use WannierFrozenType.NONE
            for metals or insulators with conduction bands:
                for WannierProjectionType.ANALYTIC/RANDOM, use WannierFrozenType.ENERGY_FIXED
                for WannierProjectionType.ATOMIC_PROJECTORS_QE/OPENMX, use WannierFrozenType.FIXED_PLUS_PROJECTABILITY
                for WannierProjectionType.SCDM, use WannierFrozenType.NONE
        :param maximal_localisation: if true do maximal localisation of Wannier functions.
        :param exclude_semicores: if True do not Wannierise semicore states.
        :param plot_wannier_functions: if True plot Wannier functions as xsf files.
        :param retrieve_hamiltonian: if True retrieve Wannier Hamiltonian.
        :param retrieve_matrices: if True retrieve amn/mmn/eig/chk/spin files.
        :param print_summary: if True print a summary of key input parameters
        :param summary: A dict containing key input parameters and can be printed out
        when the `get_builder_from_protocol` returns, to let user have a quick check of the
        generated inputs. Since in python dict is pass-by-reference, the input dict can be
        modified in the method and used by the invoking function. This allows printing the
        summary only by the last overriding method.
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.pseudo import (
            get_pseudo_orbitals,
            get_semicore_list,
        )
        from aiida_wannier90_workflows.utils.workflows.builder.projections import (
            guess_wannier_projection_types,
        )
        from aiida_wannier90_workflows.utils.workflows.builder.submit import check_codes

        codes = check_codes(codes)
        type_check(electronic_type, ElectronicType)
        type_check(spin_type, SpinType)
        type_check(projection_type, WannierProjectionType)
        if disentanglement_type:
            type_check(disentanglement_type, WannierDisentanglementType)
        if frozen_type:
            type_check(frozen_type, WannierFrozenType)

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(f"electronic type `{electronic_type}` is not supported.")

        if spin_type not in [SpinType.NONE, SpinType.SPIN_ORBIT]:
            raise NotImplementedError(f"spin type `{spin_type}` is not supported.")

        if initial_magnetic_moments and spin_type != SpinType.COLLINEAR:
            raise ValueError(f"`initial_magnetic_moments` is specified but spin type `{spin_type}` is incompatible.")

        (
            projection_type,
            disentanglement_type,
            frozen_type,
        ) = guess_wannier_projection_types(
            electronic_type=electronic_type,
            projection_type=projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
        )

        if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_OPENMX:
            if external_projectors_path is None:
                raise ValueError(f"Must specify `external_projectors_path` when using {projection_type}")
            type_check(external_projectors_path, str)

        protocol_overrides = cls.get_protocol_overrides()

        if overrides is None:
            overrides = {}

        if plot_wannier_functions:
            overrides = recursive_merge(protocol_overrides["plot_wannier_functions"], overrides)

        if retrieve_hamiltonian:
            overrides = recursive_merge(protocol_overrides["retrieve_hamiltonian"], overrides)

        if retrieve_matrices:
            overrides = recursive_merge(protocol_overrides["retrieve_matrices"], overrides)

        if pseudo_family is None:
            if spin_type == SpinType.SPIN_ORBIT:
                pseudo_family = "PseudoDojo/0.4/PBE/FR/standard/upf"
            else:
                pseudo_family = (
                    pseudo_family
                    or Wannier90BaseWorkChain.get_protocol_inputs(protocol=protocol)["meta_parameters"]["pseudo_family"]
                )

        # PwBaseWorkChain.get_builder_from_protocol() does not support SOC directly.
        spin_orbit_coupling = spin_type == SpinType.SPIN_ORBIT
        if spin_type == SpinType.NON_COLLINEAR:
            overrides = recursive_merge(protocol_overrides["spin_noncollinear"], overrides)
            pw_spin_type = SpinType.NONE
        elif spin_type == SpinType.SPIN_ORBIT:
            overrides = recursive_merge(protocol_overrides["spin_orbit"], overrides)
            pw_spin_type = SpinType.NONE
        else:
            pw_spin_type = spin_type

        inputs = cls.get_protocol_inputs(protocol=protocol, overrides=overrides)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs.get("clean_workdir"))

        wannier_overrides = inputs.get("wannier90", {})
        wannier_overrides.setdefault("meta_parameters", {})
        wannier_overrides["meta_parameters"].setdefault("exclude_semicore", exclude_semicore)
        wannier_builder = Wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["wannier90"],
            structure=structure,
            protocol=protocol,
            overrides=wannier_overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
            projection_type=projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
            pseudo_family=pseudo_family,
        )
        wannier_builder["wannier90"].pop("structure", None)
        wannier_builder.pop("clean_workdir", None)
        builder.wannier90 = wannier_builder._inputs(prune=True)

        scf_overrides = inputs.get("scf", {})
        scf_overrides["pseudo_family"] = pseudo_family
        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=codes["pw"],
            structure=structure,
            protocol=protocol,
            overrides=scf_overrides,
            electronic_type=electronic_type,
            spin_type=pw_spin_type,
        )
        scf_builder["pw"].pop("structure", None)
        scf_builder.pop("clean_workdir", None)
        builder.scf = scf_builder._inputs(prune=True)

        nscf_overrides = inputs.get("nscf", {})
        nscf_overrides["pseudo_family"] = pseudo_family

        num_bands = wannier_builder["wannier90"]["parameters"]["num_bands"]
        exclude_bands = wannier_builder["wannier90"]["parameters"].get_dict().get("exclude_bands", [])
        nscf_overrides["pw"]["parameters"]["SYSTEM"]["nbnd"] = num_bands + len(exclude_bands)

        nscf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=codes["pw"],
            structure=structure,
            protocol=protocol,
            overrides=nscf_overrides,
            electronic_type=electronic_type,
            spin_type=pw_spin_type,
        )
        # Use the explicit k-point list generated by the Wannier builder.
        nscf_builder.pop("kpoints_distance", None)
        nscf_builder.kpoints = wannier_builder["wannier90"]["kpoints"]

        nscf_builder["pw"].pop("structure", None)
        nscf_builder.pop("clean_workdir", None)
        builder.nscf = nscf_builder._inputs(prune=True)

        if projection_type == WannierProjectionType.SCDM:
            run_projwfc = True
        else:
            if frozen_type == WannierFrozenType.ENERGY_AUTO:
                run_projwfc = True
            else:
                run_projwfc = False

        if run_projwfc:
            projwfc_overrides = inputs.get("projwfc", {})
            projwfc_builder = ProjwfcBaseWorkChain.get_builder_from_protocol(
                code=codes["projwfc"], protocol=protocol, overrides=projwfc_overrides
            )
            projwfc_builder.pop("clean_workdir", None)
            builder.projwfc = projwfc_builder._inputs(prune=True)

        exclude_projectors = None
        if exclude_semicore:
            pseudo_orbitals = get_pseudo_orbitals(builder["scf"]["pw"]["pseudos"])
            exclude_projectors = get_semicore_list(structure, pseudo_orbitals, spin_orbit_coupling)

        pw2wannier90_overrides = inputs.get("pw2wannier90", {})
        pw2wannier90_builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["pw2wannier90"],
            protocol=protocol,
            overrides=pw2wannier90_overrides,
            electronic_type=electronic_type,
            projection_type=projection_type,
            exclude_projectors=exclude_projectors,
            external_projectors_path=external_projectors_path,
        )
        pw2wannier90_builder.pop("clean_workdir", None)
        builder.pw2wannier90 = pw2wannier90_builder._inputs(prune=True)

        if summary is None:
            summary = {}
        summary["Formula"] = structure.get_formula()
        summary["PseudoFamily"] = pseudo_family
        summary["ElectronicType"] = electronic_type.name
        summary["SpinType"] = spin_type.name
        summary["WannierProjectionType"] = projection_type.name
        summary["WannierDisentanglementType"] = disentanglement_type.name
        summary["WannierFrozenType"] = frozen_type.name

        params = builder["wannier90"]["wannier90"]["parameters"].get_dict()
        summary["num_bands"] = params["num_bands"]
        summary["num_wann"] = params["num_wann"]
        if "exclude_bands" in params:
            summary["exclude_bands"] = params["exclude_bands"]
        summary["mp_grid"] = params["mp_grid"]

        notes = summary.get("notes", [])
        summary["notes"] = notes

        if print_summary:
            cls.print_summary(summary)

        return builder

    @classmethod
    def print_summary(cls, summary: ty.Dict) -> None:
        """Try to pretty print the summary when the `get_builder_from_protocol` returns."""
        notes = summary.pop("notes", [])

        print("Summary of key input parameters:")
        for key, val in summary.items():
            print(f"  {key}: {val}")
        print("")

        if len(notes) == 0:
            return

        print("Notes:")
        for note in notes:
            print(f"  * {note}")

    def setup(self) -> None:
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure

        if not self.should_run_scf():
            if self.should_run_nscf():
                self.ctx.current_folder = self.inputs["nscf"]["pw"]["parent_folder"]
            elif self.should_run_projwfc():
                self.ctx.current_folder = self.inputs["projwfc"]["projwfc"]["parent_folder"]
            else:
                self.ctx.current_folder = self.inputs["pw2wannier90"]["pw2wannier90"]["parent_folder"]

    def should_run_scf(self) -> bool:
        """If the `scf` input namespace is specified, run the scf workchain."""
        return "scf" in self.inputs

    def run_scf(self):
        """Run the `PwBaseWorkChain` in scf mode on the current structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="scf"))
        inputs.pw.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = "scf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in scf mode")

        return ToContext(workchain_scf=running)

    def inspect_scf(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(f"scf {workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_run_nscf(self) -> bool:
        """If the `nscf` input namespace is specified, run the nscf workchain."""
        return "nscf" in self.inputs

    def run_nscf(self):
        """Run the PwBaseWorkChain in nscf mode."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="nscf"))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = "nscf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in nscf mode")

        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `PwBaseWorkChain` for the nscf run successfully finished."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(f"nscf {workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_run_projwfc(self) -> bool:
        """If the `projwfc` input namespace is specified, run the projwfc calculation."""
        return "projwfc" in self.inputs and not self.should_fit_cwf_parameters()

    def run_projwfc(self):
        """Run the projwfc step."""
        inputs = AttributeDict(self.exposed_inputs(ProjwfcBaseWorkChain, namespace="projwfc"))
        inputs.projwfc.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = "projwfc"

        inputs = prepare_process_inputs(ProjwfcBaseWorkChain, inputs)
        running = self.submit(ProjwfcBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_projwfc=running)

    def inspect_projwfc(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `ProjwfcCalculation` for the projwfc run successfully finished."""
        workchain = self.ctx.workchain_projwfc

        if not workchain.is_finished_ok:
            self.report(f"{workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

    def prepare_wannier90_pp_inputs(self):  # pylint: disable=too-many-statements
        """Prepare the Wannier90 post-processing inputs before submission.

        This method is called at runtime so that dynamic quantities such as the Fermi
        energy can be added after earlier calculations have finished. Derived classes
        may override this method to further modify the inputs.
        """
        from aiida_wannier90_workflows.utils.workflows.pw import (
            get_fermi_energy,
            get_fermi_energy_from_nscf,
        )

        base_inputs = AttributeDict(self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90"))
        inputs = base_inputs["wannier90"]
        inputs.structure = self.ctx.current_structure
        parameters = inputs.parameters.get_dict()

        if "workchain_scf" in self.ctx:
            scf_output_parameters = self.ctx.workchain_scf.outputs.output_parameters
            fermi_energy = get_fermi_energy(scf_output_parameters)
        elif "workchain_nscf" in self.ctx:
            fermi_energy = get_fermi_energy_from_nscf(self.ctx.workchain_nscf)
        else:
            if "fermi_energy" in parameters:
                fermi_energy = parameters["fermi_energy"]
            else:
                raise ValueError("Cannot retrieve Fermi energy from scf or nscf output")
        parameters["fermi_energy"] = fermi_energy

        if self.should_fit_cwf_parameters():
            parameters["auto_projections"] = True
            for key in (
                "dis_froz_min",
                "dis_froz_max",
                "dis_win_min",
                "dis_win_max",
            ):
                parameters.pop(key, None)
            inputs.pop("projections", None)
            base_inputs.pop("guiding_centres_projections", None)

        inputs.parameters = orm.Dict(parameters)

        if "settings" in inputs:
            settings = inputs["settings"].get_dict()
        else:
            settings = {}
        settings["postproc_setup"] = True
        inputs["settings"] = settings

        # Do not stash files in postproc mode, otherwise a RemoteStashFolderData
        # may appear in the outputs.
        inputs["metadata"]["options"].pop("stash", None)

        base_inputs["wannier90"] = inputs

        if base_inputs["shift_energy_windows"] and "bands" not in base_inputs:
            if "workchain_scf" in self.ctx:
                output_band = self.ctx.workchain_scf.outputs.output_band
            elif "workchain_nscf" in self.ctx:
                output_band = self.ctx.workchain_nscf.outputs.output_band
            else:
                raise ValueError("No output scf or nscf bands")
            base_inputs.bands = output_band

        if base_inputs["auto_energy_windows"]:
            if "bands" not in base_inputs:
                base_inputs.bands = self.ctx.workchain_projwfc.outputs.bands
            if "bands_projections" not in base_inputs:
                base_inputs.bands_projections = self.ctx.workchain_projwfc.outputs.projections

        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90_pp(self):
        """Run the Wannier90 post-processing step."""
        inputs = self.prepare_wannier90_pp_inputs()
        inputs["metadata"] = {"call_link_label": "wannier90_pp"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in postproc mode")

        return ToContext(workchain_wannier90_pp=running)

    def inspect_wannier90_pp(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `Wannier90Calculation` for the postproc run successfully finished."""
        workchain = self.ctx.workchain_wannier90_pp

        if not workchain.is_finished_ok:
            self.report(f"wannier90 postproc {workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90PP

    def prepare_pw2wannier90_inputs(self):
        """Prepare the inputs of `Pw2wannier90BaseWorkChain` before submission.

        This method is called at runtime, so it can dynamically add or modify inputs
        based on outputs of previous calculations, for example to add bands and
        projections for calculating SCDM parameters from projectability.
        """
        base_inputs = AttributeDict(self.exposed_inputs(Pw2wannier90BaseWorkChain, namespace="pw2wannier90"))
        inputs = base_inputs["pw2wannier90"]
        parameters = inputs.parameters.get_dict().get("inputpp", {})

        if self.should_fit_cwf_parameters():
            for key in (
                "scdm_proj",
                "scdm_entanglement",
                "scdm_mu",
                "scdm_sigma",
            ):
                parameters.pop(key, None)

            parameters["atom_proj"] = True

            if "settings" in inputs:
                settings = inputs.settings.get_dict()
            else:
                settings = {}

            retrieve_list = list(settings.get("additional_retrieve_list", []))
            for filename in ("aiida.amn", "aiida.eig"):
                if filename not in retrieve_list:
                    retrieve_list.append(filename)
            settings["additional_retrieve_list"] = retrieve_list
            inputs.settings = orm.Dict(settings)
            inputs.parameters = orm.Dict({"inputpp": parameters})

        scdm_proj = parameters.get("scdm_proj", False)
        scdm_entanglement = parameters.get("scdm_entanglement", None)
        scdm_mu = parameters.get("scdm_mu", None)
        scdm_sigma = parameters.get("scdm_sigma", None)

        fit_scdm = scdm_proj and scdm_entanglement == "erfc" and (scdm_mu is None or scdm_sigma is None)

        if fit_scdm:
            if "workchain_projwfc" not in self.ctx:
                raise ValueError("Needs to run projwfc for SCDM projection")
            base_inputs["bands"] = self.ctx.workchain_projwfc.outputs.bands
            base_inputs["bands_projections"] = self.ctx.workchain_projwfc.outputs.projections

        inputs["parent_folder"] = self.ctx.current_folder
        inputs["nnkp_file"] = self.ctx.workchain_wannier90_pp.outputs.nnkp_file

        base_inputs["pw2wannier90"] = inputs

        return base_inputs

    def run_pw2wannier90(self):
        """Run the pw2wannier90 step."""
        inputs = self.prepare_pw2wannier90_inputs()
        inputs.metadata.call_link_label = "pw2wannier90"

        inputs = prepare_process_inputs(Pw2wannier90BaseWorkChain, inputs)
        running = self.submit(Pw2wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_pw2wannier90=running)

    def inspect_pw2wannier90(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the Pw2wannier90BaseWorkChain for the pw2wannier90 run successfully finished."""
        workchain = self.ctx.workchain_pw2wannier90

        if not workchain.is_finished_ok:
            self.report(f"{workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_fit_cwf_parameters(self):
        """Return whether Closest Wannier parameters should be fitted."""
        return self.inputs.auto_cwf_parameters.value

    def _get_cwf_dimensions_from_amn(self):
        """Read num_bands and num_wann from the retrieved aiida.amn file."""
        from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

        last_calc = get_last_calcjob(self.ctx.workchain_pw2wannier90)
        if last_calc is None or "retrieved" not in last_calc.outputs:
            return self.exit_codes.ERROR_CWF_FILES_MISSING

        try:
            amn_content = last_calc.outputs.retrieved.get_object_content("aiida.amn")
        except (IOError, OSError, KeyError):
            self.report("cannot read `aiida.amn` to determine CWF dimensions")
            return self.exit_codes.ERROR_CWF_FILES_MISSING

        lines = amn_content.splitlines()

        if len(lines) < 2:
            self.report("`aiida.amn` is malformed")
            return self.exit_codes.ERROR_CWF_FITTING_FAILED

        try:
            num_bands, _num_kpoints, num_wann = map(int, lines[1].split()[:3])
        except (ValueError, IndexError) as exc:
            self.report(f"failed to parse `aiida.amn`: {exc}")
            return self.exit_codes.ERROR_CWF_FITTING_FAILED

        return {
            "num_bands": num_bands,
            "num_wann": num_wann,
        }

    def fit_cwf_parameters(self):  # pylint: disable=inconsistent-return-statements
        """Fit Closest Wannier parameters from the retrieved pw2wannier90 files."""
        from aiida_wannier90_workflows.utils.cwf import fit_cwf_parameters_from_contents
        from aiida_wannier90_workflows.utils.workflows.pw import (
            get_fermi_energy,
            get_fermi_energy_from_nscf,
        )
        from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

        last_calc = get_last_calcjob(self.ctx.workchain_pw2wannier90)
        if last_calc is None or "retrieved" not in last_calc.outputs:
            self.report("cannot fit CWF parameters because the pw2wannier90 retrieved " "folder is unavailable")
            return self.exit_codes.ERROR_CWF_FILES_MISSING

        try:
            eig_content = last_calc.outputs.retrieved.get_object_content("aiida.eig")
            amn_content = last_calc.outputs.retrieved.get_object_content("aiida.amn")
        except (IOError, OSError, KeyError):
            self.report("cannot fit CWF parameters because `aiida.eig` or `aiida.amn` " "was not retrieved")
            return self.exit_codes.ERROR_CWF_FILES_MISSING

        if "workchain_scf" in self.ctx:
            scf_output_parameters = self.ctx.workchain_scf.outputs.output_parameters
            fermi_energy = get_fermi_energy(scf_output_parameters)
        elif "workchain_nscf" in self.ctx:
            fermi_energy = get_fermi_energy_from_nscf(self.ctx.workchain_nscf)
        else:
            fermi_energy = self.inputs.wannier90.wannier90.parameters.get_dict().get("fermi_energy")

        try:
            parameters = fit_cwf_parameters_from_contents(
                eig_content=eig_content,
                amn_content=amn_content,
                sigma_factor=self.inputs.cwf_sigma_factor.value,
                delta=self.inputs.cwf_delta.value,
                fermi_energy=fermi_energy,
            )
        except (RuntimeError, TypeError, ValueError) as exception:
            self.report(f"CWF fitting failed: {exception}")
            return self.exit_codes.ERROR_CWF_FITTING_FAILED

        dims = self._get_cwf_dimensions_from_amn()
        if isinstance(dims, ExitCode):
            return dims

        self.ctx.cwf_parameters = orm.Dict(dict=parameters)
        self.ctx.cwf_dimensions = orm.Dict(dict=dims)

        self.report("fitted CWF parameters: " + ", ".join(f"{key}={value:.8f}" for key, value in parameters.items()))
        self.report("CWF dimensions from aiida.amn: " f"num_bands={dims['num_bands']}, num_wann={dims['num_wann']}")

    def prepare_wannier90_inputs(self):  # pylint: disable=too-many-statements
        """Prepare the final Wannier90 inputs before submission.

        This method is called at runtime to add dynamic values and to reuse the
        corrected inputs generated during the post-processing step.
        """
        from copy import deepcopy

        from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

        base_inputs = AttributeDict(self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90"))

        # Disable energy-window shifting here because it has already been handled
        # in the post-processing step.
        base_inputs.pop("shift_energy_windows", None)
        base_inputs.pop("auto_energy_windows", None)
        base_inputs.pop("auto_energy_windows_threshold", None)
        base_inputs.pop("bands", None)
        base_inputs.pop("bands_projections", None)

        inputs = base_inputs["wannier90"]

        # Save the stash settings because they were removed in postproc mode.
        stash = None
        if "stash" in inputs["metadata"]["options"]:
            stash = deepcopy(inputs["metadata"]["options"]["stash"])

        last_calc = get_last_calcjob(self.ctx.workchain_wannier90_pp)

        # Reuse the corrected inputs from the postproc Wannier90 calculation.
        # For example, `kmesh_tol` may have been adjusted there.
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        inputs["remote_input_folder"] = self.ctx.current_folder

        if "settings" in inputs:
            settings = inputs.settings.get_dict()
        else:
            settings = {}
        settings["postproc_setup"] = False
        inputs.settings = settings

        if self.should_fit_cwf_parameters():
            parameters = inputs.parameters.get_dict()
            parameters["auto_projections"] = True
            parameters["guiding_centres"] = False
            parameters["num_iter"] = 0
            parameters["dis_num_iter"] = 0
            parameters["use_cwf_method"] = True
            parameters["cwf_delta"] = self.inputs.cwf_delta.value

            for key in (
                # Projection-related keys
                "projections",
                # Disentanglement and energy-window related keys
                "dis_froz_min",
                "dis_froz_max",
                "dis_win_min",
                "dis_win_max",
                # Max-localisation / disentanglement iteration control
                "conv_tol",
                "conv_window",
                "dis_conv_tol",
                "num_cg_steps",
                # SCDM-related keys
                "scdm_mu",
                "scdm_sigma",
                "scdm_entanglement",
            ):
                parameters.pop(key, None)

            if "cwf_parameters" not in self.ctx:
                return self.exit_codes.ERROR_CWF_FITTING_FAILED

            if "cwf_dimensions" not in self.ctx:
                return self.exit_codes.ERROR_CWF_FITTING_FAILED

            for key, value in self.ctx.cwf_parameters.get_dict().items():
                if key not in parameters:
                    parameters[key] = value

            cwf_dimensions = self.ctx.cwf_dimensions.get_dict()
            parameters["num_bands"] = cwf_dimensions["num_bands"]
            parameters["num_wann"] = cwf_dimensions["num_wann"]

            inputs.pop("projections", None)
            base_inputs.pop("guiding_centres_projections", None)
            inputs.parameters = orm.Dict(parameters)

        if stash:
            options = deepcopy(inputs["metadata"]["options"])
            options["stash"] = stash
            inputs["metadata"]["options"] = options

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90(self):
        """Run the final Wannier90 step for MLWF."""
        inputs = self.prepare_wannier90_inputs()
        if isinstance(inputs, ExitCode):
            return inputs

        inputs["metadata"] = {"call_link_label": "wannier90"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_wannier90=running)

    def inspect_wannier90(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `Wannier90BaseWorkChain` for the final Wannier90 run successfully finished."""
        workchain = self.ctx.workchain_wannier90

        if not workchain.is_finished_ok:
            self.report(f"{workchain.process_label} failed with exit status {workchain.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder

    def results(self):  # pylint: disable=inconsistent-return-statements
        """Attach the desired output nodes directly as outputs of the workchain."""
        if "workchain_scf" in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.workchain_scf, PwBaseWorkChain, namespace="scf"))

        if "workchain_nscf" in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace="nscf"))

        if "workchain_projwfc" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_projwfc,
                    ProjwfcBaseWorkChain,
                    namespace="projwfc",
                )
            )

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_pw2wannier90,
                Pw2wannier90BaseWorkChain,
                namespace="pw2wannier90",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_wannier90_pp,
                Wannier90BaseWorkChain,
                namespace="wannier90_pp",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_wannier90,
                Wannier90BaseWorkChain,
                namespace="wannier90",
            )
        )
        if "cwf_parameters" in self.ctx:
            self.out("cwf_parameters", self.ctx.cwf_parameters)

        result = self.sanity_check()
        if result:
            return result

        self.report(f"{self.get_name()} successfully completed")

    def sanity_check(self):  # pylint: disable=inconsistent-return-statements
        """Run sanity checks for final outputs."""
        from aiida_wannier90_workflows.utils.pseudo import (
            get_number_of_electrons,
            get_number_of_projections,
        )

        p2w_params = self.ctx.workchain_pw2wannier90.inputs["pw2wannier90"]["parameters"].get_dict()["inputpp"]
        atom_proj = p2w_params.get("atom_proj", False)
        atom_proj_ext = p2w_params.get("atom_proj_ext", False)
        if atom_proj and atom_proj_ext:
            return

        check_num_projs = True
        if self.should_run_scf():
            pseudos = self.inputs["scf"]["pw"]["pseudos"]
        elif self.should_run_nscf():
            pseudos = self.inputs["nscf"]["pw"]["pseudos"]
        else:
            check_num_projs = False

        if check_num_projs:
            args = {
                "structure": self.ctx.current_structure,
                # `self.inputs['scf']['pw']['pseudos']` is an AttributesFrozendict,
                # so convert it to a plain dict before passing it on.
                "pseudos": dict(pseudos),
            }
            if "workchain_projwfc" in self.ctx:
                num_proj = len(self.ctx.workchain_projwfc.outputs["projections"].get_orbitals())
                params = self.ctx.workchain_wannier90.inputs["wannier90"]["parameters"].get_dict()
                spin_orbit_coupling = params.get("spinors", False)
                number_of_projections = get_number_of_projections(**args, spin_orbit_coupling=spin_orbit_coupling)
                if number_of_projections != num_proj:
                    self.report(f"number of projections {number_of_projections} != projwfc.x output {num_proj}")
                    return self.exit_codes.ERROR_SANITY_CHECK_FAILED

        check_num_elecs = check_num_projs
        if "workchain_scf" in self.ctx:
            num_elec = self.ctx.workchain_scf.outputs["output_parameters"]["number_of_electrons"]
        elif "workchain_nscf" in self.ctx:
            num_elec = self.ctx.workchain_nscf.outputs["output_parameters"]["number_of_electrons"]
        else:
            check_num_elecs = False

        if check_num_elecs:
            number_of_electrons = get_number_of_electrons(**args)
            if number_of_electrons != num_elec:
                self.report(f"number of electrons {number_of_electrons} != QE output {num_elec}")
                return self.exit_codes.ERROR_SANITY_CHECK_FAILED

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True`."""
        super().on_terminated()

        if not self.inputs.clean_workdir:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
