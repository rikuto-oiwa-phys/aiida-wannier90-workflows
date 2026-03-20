#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""
launch_cw_atom_proj.py

目的:
- 過去の Wannier90OptimizeWorkChain / Wannier90BandsWorkChain の入力を再利用
- 元 WorkChain に応じて Wannier90OptimizeWorkChain / Wannier90BandsWorkChain を使い分ける
- Closest Wannier (CWF) を有効化
- seekpath は完全無効
- NSCF の structure は常に元 WorkChain 入力の structure を使う
- 親 remote は第2引数リストから structure + description マッチで SCF 親を選ぶ
- SCF 親は <remote>/<outdir>/<prefix>.save/data-file-schema.xml の存在で検証（stash優先）
- NSCF の parent_folder はその親 SCF に設定
- bands_kpoints / explicit_kpath / explicit_kpath_labels は保持
- kpoint_path は一切使わない
- wannier90.wannier90.kpoints は必ず設定
- QE 系の parallelization 入力は全削除
- QE 系 settings の CMDLINE は空に固定
- MPI は NSCF / pw2wannier90 / projwfc = 2 固定
- wannier90 = 1 固定
- pw2wannier90 parameters は v3/QE7 仕様に整形
- Optimize 専用入力は削除
- CWF 用に auto_cwf_parameters / cwf_delta / auto_projections / use_cwf_method などを設定

使い方:
  USE_SUBMIT=1 python launch_cw_atom_proj.py <wannier_list.txt> <scf_list.txt>

  USE_SUBMIT=1 python launch_cw_atom_proj.py list_of_calculations_for_200_structures_wannier_atom_proj_test.txt  list_of_calculations_for_200_structures_pwbands_hemulen.txt

  USE_SUBMIT=1 python launch_cw_atom_proj.py list_of_calculations_for_200_structures_wannier_atom_proj_1.txt list_of_calculations_for_200_structures_pwbands_hemulen.txt

  USE_SUBMIT=1 python launch_cw_atom_proj.py list_of_calculations_for_200_structures_wannier_atom_proj_1_hemulen_error.txt list_of_calculations_for_200_structures_pwbands_hemulen.txt
"""

import os
import sys
import re
import difflib
from collections import defaultdict
from typing import List, Tuple, Optional, Set

from aiida import load_profile
from aiida.engine import run, submit
from aiida.common.links import LinkType
from aiida.orm import (
    Bool,
    Float,
    Dict,
    Node,
    Code,
    RemoteData,
    CalcJobNode,
    WorkChainNode,
    load_node,
    load_code,
    KpointsData,
    Dict as AiiDADict,
    List as AiiDAList,
    StructureData,
)
from aiida.plugins import WorkflowFactory
from aiida_wannier90_workflows.utils.kpoints import get_kpoints_from_bands

Wannier90BandsWorkChain = WorkflowFactory("wannier90_workflows.bands")
Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

# ==========================
# 設定・定数
# ==========================

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
USE_SUBMIT = os.environ.get("USE_SUBMIT", "0") == "1"

CODE_MAP_HEMULEN = {
    "quantumespresso.pw": "pw-7.3-intel@hemulen-GroupA",
    "quantumespresso.pw2wannier90": "pw2wan-7.3-intel@hemulen-GroupA",
    "wannier90.wannier90": "wannier90-develop-intel@hemulen-GroupA",
    "quantumespresso.projwfc": "projwfc-7.3-intel@hemulen-GroupA",
}

FIXED_MPI = 8
W90_MPI = 1
DEFAULT_WALLCLOCK = 2 * 3600


# ==========================
# 汎用ユーティリティ
# ==========================


def _walk_dict(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield d, k, v
            yield from _walk_dict(v)


def rebind_codes(inputs, code_map):
    """inputs 内の Code を endpoint で判定し、hemulen 用 Code に差し替える。"""

    def _as_code(identifier):
        if isinstance(identifier, Code):
            return identifier
        try:
            return load_code(identifier)
        except Exception:
            return Code.get(int(identifier))

    resolved = {ep: _as_code(label) for ep, label in code_map.items()}

    for parent, key, val in _walk_dict(inputs):
        if isinstance(val, Code):
            ep = val.default_calc_job_plugin
            if ep in resolved:
                parent[key] = resolved[ep]


def ensure_resources_under(
    namespace: dict,
    mpi: int = FIXED_MPI,
    wallclock_s: int = DEFAULT_WALLCLOCK,
    queue_name=None,
    *,
    force: bool = False,
):
    """metadata.options.resources / max_wallclock_seconds をセット。"""
    md = namespace.setdefault("metadata", {})
    opts = md.setdefault("options", {})

    if force:
        opts["resources"] = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": int(mpi),
        }
    else:
        res = opts.setdefault("resources", {})
        res.setdefault("num_machines", 1)
        res.setdefault("num_mpiprocs_per_machine", int(mpi))

    opts["max_wallclock_seconds"] = int(wallclock_s)
    if queue_name:
        opts["queue_name"] = queue_name


def reconstruct_inputs_from_links(proc_node: Node):
    """INPUT_CALC / INPUT_WORK リンクから inputs をネスト辞書として再構成。"""
    nested = lambda: defaultdict(nested)
    root = nested()

    for link in proc_node.base.links.get_incoming().all():
        if link.link_type not in (LinkType.INPUT_WORK, LinkType.INPUT_CALC):
            continue
        label = link.link_label
        parts = label.split("__") if "__" in label else label.split(".")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = link.node

    def _to_plain(d):
        if isinstance(d, defaultdict):
            return {k: _to_plain(v) for k, v in d.items()}
        return d

    return _to_plain(root)


def parse_list_file(path):
    """list_of_calculations_* 形式のリストをパース。"""
    records = []
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if not line.startswith("  "):
                current_section = line.strip()
                continue

            m = UUID_RE.search(line)
            if not m:
                continue

            uuid = m.group(0)
            trimmed = line.strip()
            fields = trimmed.split()

            try:
                idx = fields.index(uuid)
                structure = fields[0] if fields else ""
                process_label = fields[idx + 1] if idx + 1 < len(fields) else ""
            except ValueError:
                structure = trimmed.split()[0] if trimmed else ""
                process_label = ""

            description = ""
            if process_label:
                tail = trimmed[trimmed.find(uuid) + len(uuid) :].strip()
                if tail.startswith(process_label):
                    description = tail[len(process_label) :].strip()

            records.append(
                {
                    "section": current_section or "",
                    "structure": structure,
                    "uuid": uuid,
                    "process_label": process_label,
                    "description": description,
                }
            )

    return records


def preview(obj, indent=0, max_depth=3):
    """デバッグ用 preview。"""
    pad = "  " * indent
    if indent > max_depth:
        print(pad + "...")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(pad + f"{k}:")
            preview(v, indent + 1, max_depth)
    else:
        try:
            if isinstance(obj, Node):
                print(pad + f"<{obj.__class__.__name__} pk={obj.pk}>")
            else:
                print(pad + repr(obj))
        except Exception:
            print(pad + f"<{type(obj).__name__}>")


# ==========================
# SCF / NSCF 判定・親 SCF 選定
# ==========================


def _collect_descendants(root: Node) -> List[Node]:
    out: List[Node] = []
    seen: Set[int] = set()
    stack = [root]

    while stack:
        n = stack.pop()
        for ln in n.base.links.get_outgoing().all():
            if ln.link_type not in (LinkType.CALL_CALC, LinkType.CALL_WORK):
                continue
            ch = ln.node
            if ch.pk in seen:
                continue
            seen.add(ch.pk)
            out.append(ch)
            stack.append(ch)

    return out


def _calc_is_pw_scf(calc: CalcJobNode) -> bool:
    try:
        if "quantumespresso.pw" not in (calc.process_type or ""):
            return False
        p = calc.inputs.parameters.get_dict()
        ctrl = p.get("CONTROL") or p.get("control") or {}
        return ctrl.get("calculation") == "scf"
    except Exception:
        return False


def _get_pw_prefix_outdir_from_calc(calc: CalcJobNode) -> Tuple[str, str]:
    p = calc.inputs.parameters.get_dict() if hasattr(calc.inputs, "parameters") else {}
    ctrl = p.get("CONTROL") or p.get("control") or {}
    prefix = ctrl.get("prefix", "aiida")
    outdir = ctrl.get("outdir", "./out")
    return prefix, outdir


def _remote_has_qe_xml(remote: RemoteData, prefix: str, outdir: str, *, debug: bool = False) -> bool:
    """RemoteData 直下で prefix.save/data-file-schema.xml を探す。"""
    try:
        base = remote.get_remote_path().rstrip("/")
        cand_dirs = []

        od = (outdir or "").strip()
        if od.endswith("/"):
            od = od[:-1]

        if od.startswith("/"):
            cand_dirs.append(od)
        else:
            trimmed = od
            if trimmed.startswith("./"):
                trimmed = trimmed[2:]
            trimmed = trimmed.lstrip("/")
            if trimmed:
                cand_dirs.append(f"{base}/{trimmed}")
            if od:
                cand_dirs.append(f"{base}/{od.lstrip('./')}")
            cand_dirs.append(base)

        candidates = [f"{d}/{prefix}.save/data-file-schema.xml" for d in cand_dirs]

        with remote.computer.get_transport() as t:
            for p in candidates:
                ok = t.path_exists(p)
                if debug:
                    print(f"    [CHECK] {p} -> {ok}")
                if ok:
                    return True
        return False
    except Exception as e:
        if debug:
            print(f"    [CHECK] transport error: {e}")
        return False


def _make_remote_from_stash(scf_calc: CalcJobNode) -> Optional[RemoteData]:
    """親 PwBase の remote_stash から RemoteData を合成。"""
    parent_wc = None
    try:
        for ln in scf_calc.base.links.get_incoming().all():
            if isinstance(ln.node, WorkChainNode) and "PwBaseWorkChain" in (ln.node.process_label or ""):
                parent_wc = ln.node
                break
    except Exception:
        parent_wc = None

    if parent_wc is None:
        return None

    stash = None
    try:
        stash = parent_wc.base.links.get_outgoing().get_node_by_label("remote_stash")
    except Exception:
        stash = None
    if stash is None:
        return None

    try:
        basepath = stash.base.attributes.all.get("target_basepath", None)
    except Exception:
        basepath = None
    if not basepath:
        return None

    comp = None
    try:
        comp = scf_calc.outputs.remote_folder.computer
    except Exception:
        pass
    if comp is None:
        try:
            comp = parent_wc.outputs.remote_folder.computer
        except Exception:
            pass
    if comp is None:
        return None

    rd = RemoteData(computer=comp, remote_path=basepath.rstrip("/"))
    try:
        rd.store()
    except Exception:
        pass

    return rd


def _iter_possible_remote_folders_for_scf_calc(scf_calc: CalcJobNode):
    """優先順: stash -> 自身 remote_folder -> 親 PwBase remote_folder"""
    candidates = []

    try:
        rd_stash = _make_remote_from_stash(scf_calc)
        if isinstance(rd_stash, RemoteData):
            candidates.append(rd_stash)
    except Exception:
        pass

    try:
        rf = scf_calc.outputs.remote_folder
        if isinstance(rf, RemoteData):
            candidates.append(rf)
    except Exception:
        pass

    try:
        for ln in scf_calc.base.links.get_incoming().all():
            node = ln.node
            if isinstance(node, WorkChainNode) and "PwBaseWorkChain" in (node.process_label or ""):
                try:
                    rf2 = node.outputs.remote_folder
                    if isinstance(rf2, RemoteData):
                        candidates.append(rf2)
                except Exception:
                    pass
    except Exception:
        pass

    seen, uniq = set(), []
    for r in candidates:
        if r.pk not in seen:
            uniq.append(r)
            seen.add(r.pk)
    return uniq


def _pick_parent_by_structure_and_description(scf_records, target_structure, target_description):
    """第2引数リストから structure + description マッチで SCF 親候補を選ぶ。"""
    allow_no_xml = os.environ.get("ALLOW_PARENT_WITHOUT_XML", "0") == "1"
    debug_parent = os.environ.get("DEBUG_PARENT", "0") == "1"

    cand = [r for r in scf_records if r["structure"] == target_structure]
    if not cand:
        return None

    exact = [r for r in cand if (r.get("description") or "") == (target_description or "")]
    blanks = [r for r in cand if not (r.get("description") or "")]
    rest = [r for r in cand if r not in exact and r not in blanks]

    scored = sorted(
        (
            (r, difflib.SequenceMatcher(a=(target_description or ""), b=(r.get("description") or "")).ratio())
            for r in rest
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    ordering = exact + blanks + [r for (r, _) in scored]

    for rec in ordering:
        try:
            root = load_node(rec["uuid"])
        except Exception:
            continue

        descendants = _collect_descendants(root)
        scf_calcs = [n for n in descendants if isinstance(n, CalcJobNode) and _calc_is_pw_scf(n)]
        scf_calcs.sort(key=lambda n: n.mtime or n.ctime, reverse=True)

        best_without_xml = None

        for scf_calc in scf_calcs:
            pref, outd = _get_pw_prefix_outdir_from_calc(scf_calc)
            for remote in _iter_possible_remote_folders_for_scf_calc(scf_calc):
                ok = _remote_has_qe_xml(remote, pref, outd, debug=debug_parent)
                if ok:
                    return remote, scf_calc, rec["uuid"]
                if best_without_xml is None:
                    best_without_xml = (remote, scf_calc)

        if allow_no_xml and best_without_xml is not None:
            remote, scf_calc = best_without_xml
            return remote, scf_calc, rec["uuid"]

    return None


def inherit_parent_prefix_from_scf(inputs: dict, parent_pwcalc: CalcJobNode):
    """NSCF input から CONTROL.prefix/outdir を削除。AiiDA に任せる。"""
    _ = parent_pwcalc
    nscf_pw = inputs.setdefault("nscf", {}).setdefault("pw", {})
    pnode = nscf_pw.get("parameters")
    pd = dict(pnode.get_dict()) if isinstance(pnode, AiiDADict) else {}

    for sec in ("CONTROL", "control"):
        if sec in pd and isinstance(pd[sec], dict):
            pd[sec].pop("prefix", None)
            pd[sec].pop("outdir", None)

    pd.pop("prefix", None)
    pd.pop("outdir", None)

    nscf_pw["parameters"] = AiiDADict(dict=pd)
    inputs["nscf"]["pw"] = nscf_pw
    print("  -> [NSCF] remove CONTROL.prefix/outdir (AiiDA sets them automatically)")


# ==========================
# parallelization / settings 掃除
# ==========================


def remove_qe_parallelization_inputs_everywhere(inputs: dict):
    """inputs 全体を再帰走査して parallelization 入力を削除。"""
    if not isinstance(inputs, dict):
        return

    inputs.pop("parallelization", None)

    for _, val in list(inputs.items()):
        if isinstance(val, dict):
            remove_qe_parallelization_inputs_everywhere(val)


def force_empty_settings(namespace: dict, label: str):
    """settings の CMDLINE を空に固定。"""
    namespace["settings"] = AiiDADict(dict={"CMDLINE": []})
    print(f"  -> [{label}] force settings CMDLINE = []")


def debug_print_parallel_state(inputs: dict):
    """submit 前確認用。"""
    targets = [
        ("nscf", "pw"),
        ("pw2wannier90", "pw2wannier90"),
        ("projwfc", "projwfc"),
        ("wannier90", "wannier90"),
    ]
    for p1, p2 in targets:
        try:
            ns = inputs[p1][p2]
        except Exception:
            continue

        print(f"[DEBUG] {p1}/{p2}")
        if "parallelization" in ns:
            print("    parallelization =", ns["parallelization"])
        else:
            print("    parallelization = <none>")

        s = ns.get("settings")
        if isinstance(s, AiiDADict):
            print("    settings =", s.get_dict())
        else:
            print("    settings = <none>")

        md = ns.get("metadata", {})
        print("    resources =", md.get("options", {}).get("resources"))


# ==========================
# NSCF 構造 / seekpath 無効 / Wannier90 整形
# ==========================


def force_nscf_structure_from_original(inputs: dict):
    """NSCF の structure は第1引数側元 WorkChain 入力の structure を必ず使う。"""
    orig_struct = inputs.get("structure", None)
    if isinstance(orig_struct, StructureData):
        nscf_pw = inputs.setdefault("nscf", {}).setdefault("pw", {})
        nscf_pw["structure"] = orig_struct
        print(f"  -> [STRUCT] NSCF structure forced to ORIGINAL structure pk={orig_struct.pk}")


def disable_seekpath_and_clean_artifacts(inputs: dict):
    """
    seekpath 完全無効化。
    bands_kpoints / explicit_kpath / explicit_kpath_labels は保持。
    """
    w_outer = inputs.get("wannier90", {})
    w = w_outer.get("wannier90", {})
    for key in ("use_seekpath", "kpoint_path"):
        if key in w:
            w.pop(key, None)
    inputs["wannier90"] = w_outer

    for k in ("primitive_structure", "seekpath_parameters"):
        inputs.pop(k, None)

    print("  -> [SEEKPATH] disabled and artifacts cleaned (kept bands_kpoints/explicit_* intact)")


def ensure_required_w90_kpoints(inputs: dict):
    """
    wannier90.wannier90.kpoints を必ず設定。
    優先順: NSCF mesh -> bands_kpoints -> Γ
    """
    w_outer = inputs.setdefault("wannier90", {})
    w = w_outer.setdefault("wannier90", {})

    nscf_k = inputs.get("nscf", {}).get("kpoints", None)
    if isinstance(nscf_k, KpointsData):
        w["kpoints"] = nscf_k
        print("  -> [W90.KPTS] set from NSCF mesh")
    else:
        bk = inputs.get("bands_kpoints", None)
        if isinstance(bk, KpointsData):
            w["kpoints"] = bk
            print("  -> [W90.KPTS] set from bands_kpoints (fallback)")
        else:
            gamma = KpointsData()
            gamma.set_kpoints([[0.0, 0.0, 0.0]], cartesian=False)
            w["kpoints"] = gamma
            print("  -> [W90.KPTS] fallback Γ-only (WARN)")

    w_outer["wannier90"] = w
    inputs["wannier90"] = w_outer


def sanitize_wannier90_settings(inputs: dict):
    """wannier90.settings の危険キーを除去。"""
    wns = inputs.get("wannier90", {}).get("wannier90", {})
    s = wns.get("settings")
    if isinstance(s, AiiDADict):
        d = s.get_dict()
        changed = False
        for bad in ("additional_win", "ADDITIONAL_WIN"):
            if bad in d:
                d.pop(bad, None)
                changed = True
        if changed:
            wns["settings"] = AiiDADict(dict=d)
            inputs.setdefault("wannier90", {})["wannier90"] = wns


def sanitize_wannier90_parameters(inputs: dict):
    """wannier90.parameters から不要キーを除去。"""
    wns = inputs.get("wannier90", {}).get("wannier90", {})
    pnode = wns.get("parameters")
    if not isinstance(pnode, AiiDADict):
        return

    d = dict(pnode.get_dict())
    for bad in ("amn_formatted", "eig_formatted", "mmn_formatted", "wvfn_formatted"):
        d.pop(bad, None)

    wns["parameters"] = AiiDADict(dict=d)
    inputs.setdefault("wannier90", {})["wannier90"] = wns


def ensure_wannier90_serial_by_resources_only(inputs: dict):
    """wannier90 は resources のみで 1 並列に固定。"""
    wns = inputs.setdefault("wannier90", {}).setdefault("wannier90", {})
    md = wns.setdefault("metadata", {})
    opts = md.setdefault("options", {})
    opts["resources"] = {"num_machines": 1, "num_mpiprocs_per_machine": W90_MPI}
    opts["max_wallclock_seconds"] = DEFAULT_WALLCLOCK
    inputs["wannier90"]["wannier90"] = wns


def ensure_bands_plot_true(inputs: dict):
    """bands_plot / wannier_plot を有効化。"""
    w90_ns = inputs.setdefault("wannier90", {}).setdefault("wannier90", {})
    params_node = w90_ns.get("parameters", None)
    params = params_node.get_dict() if isinstance(params_node, AiiDADict) else {}

    params["bands_plot"] = True
    params["wannier_plot"] = True
    params["wannier_plot_format"] = "xcrysden"
    params["wannier_plot_supercell"] = 3
    params["write_tb"] = True

    w90_ns["parameters"] = AiiDADict(dict=params)
    inputs["wannier90"]["wannier90"] = w90_ns


def ensure_bands_kpoints_from_reference_bands(inputs: dict):
    """Recover bands_kpoints from optimize_reference_bands when needed."""
    if "bands_kpoints" in inputs:
        return

    reference_bands = inputs.get("optimize_reference_bands")
    if reference_bands is None:
        print("  -> [WARN] no bands_kpoints and no optimize_reference_bands; seekpath may still run")
        return

    try:
        inputs["bands_kpoints"] = get_kpoints_from_bands(reference_bands)
        print("  -> [OPT] recovered bands_kpoints from optimize_reference_bands")
    except Exception as exc:
        print(f"  -> [WARN] failed to recover bands_kpoints from optimize_reference_bands: {exc}")


def enable_cwf_mode(inputs: dict):
    """Closest Wannier 用に入力を整える。"""
    inputs["auto_cwf_parameters"] = Bool(True)
    inputs["cwf_delta"] = Float(1e-12)
    inputs["optimize_disprojmax_range"] = AiiDAList(
        list=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85]
    )
    inputs["optimize_disprojmin_range"] = AiiDAList(list=[0.00, 0.01, 0.02])

    # CWF では projwfc は不要
    if "projwfc" in inputs:
        inputs.pop("projwfc", None)
        print("  -> [CWF] removed projwfc namespace")

    w90_ns = inputs.setdefault("wannier90", {}).setdefault("wannier90", {})
    params_node = w90_ns.get("parameters")
    params = params_node.get_dict() if isinstance(params_node, AiiDADict) else {}

    for key in (
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
        "scdm_mu",
        "scdm_sigma",
        "scdm_entanglement",
        "conv_tol",
        "conv_window",
        "dis_conv_tol",
        "num_cg_steps",
    ):
        params.pop(key, None)

    params["auto_projections"] = True
    params["num_iter"] = 0
    params["dis_num_iter"] = 0
    params["use_cwf_method"] = True
    params["cwf_delta"] = 1e-12

    # 必要なら固定値を使う:
    # params.setdefault("cwf_sigma_min", -1000.0)

    w90_ns["parameters"] = AiiDADict(dict=params)
    w90_ns.pop("projections", None)
    inputs["wannier90"]["wannier90"] = w90_ns

    print("  -> [CWF] enabled auto_cwf_parameters=True, cwf_delta=1e-12")


# ==========================
# pw2wannier90 / projwfc 整形
# ==========================


def sanitize_pw2wannier90_parameters(inputs: dict):
    """Pw2wannier90Calculation 用 parameters を v3/QE7 仕様に正規化。"""
    ns = inputs.get("pw2wannier90", {}).get("pw2wannier90", {})
    pnode = ns.get("parameters")
    if not isinstance(pnode, AiiDADict):
        return

    raw = dict(pnode.get_dict())
    ip = {}

    if isinstance(raw.get("INPUTPP"), dict):
        ip.update(raw.pop("INPUTPP"))
    if isinstance(raw.get("inputpp"), dict):
        ip.update(raw.pop("inputpp"))

    for k in list(raw.keys()):
        ip[k] = raw.pop(k)

    for bad in ("amn_formatted", "eig_formatted", "mmn_formatted"):
        ip.pop(bad, None)

    ns["parameters"] = AiiDADict(dict={"inputpp": ip})
    inputs.setdefault("pw2wannier90", {})["pw2wannier90"] = ns


# ==========================
# outdir 一掃
# ==========================


def _strip_outdir_anywhere(d):
    """ネストされた Dict/AiiDADict の中から outdir を消す。"""
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if isinstance(v, AiiDADict):
                dd = dict(v.get_dict())
                for sec in ("CONTROL", "control"):
                    if sec in dd and isinstance(dd[sec], dict):
                        dd[sec].pop("outdir", None)
                dd.pop("outdir", None)
                d[k] = AiiDADict(dict=dd)
            elif isinstance(v, dict):
                _strip_outdir_anywhere(v)


# ==========================
# メイン
# ==========================


def main(wannier_list_path, scf_list_path):
    load_profile()

    target_labels = {"Wannier90BandsWorkChain", "Wannier90OptimizeWorkChain"}
    wrecs = [r for r in parse_list_file(wannier_list_path) if r["process_label"] in target_labels]

    if not wrecs:
        print(f"[ERROR] No Wannier90 entries found in: {wannier_list_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(wrecs)} Wannier90 entries with UUIDs across sections.")

    scf_recs = parse_list_file(scf_list_path)
    if not scf_recs:
        print(f"[WARN] SCF list is empty: {scf_list_path}")

    submitted_rows = []
    sections_written = defaultdict(list)

    for i, rec in enumerate(wrecs, 1):
        sec = rec["section"]
        struct = rec["structure"]
        uuid = rec["uuid"]
        src_plabel = rec["process_label"]
        desc = rec["description"]

        print("\n" + "=" * 80)
        print(f"[{i}] section={sec}  structure={struct}  uuid={uuid}")
        print(f"    source process_label={src_plabel}")
        if desc:
            print(f"    description         ={desc}")

        try:
            node = load_node(uuid)
        except Exception as e:
            print(f"  -> [SKIP] load_node failed: {e}")
            continue

        if not isinstance(node, (WorkChainNode, CalcJobNode)):
            print(f"  -> [SKIP] Node is not a process node: {type(node)}")
            continue

        if src_plabel == "Wannier90OptimizeWorkChain":
            proc_cls = Wannier90OptimizeWorkChain
        else:
            proc_cls = Wannier90BandsWorkChain
        inputs = reconstruct_inputs_from_links(node)

        if os.environ.get("DEBUG_INPUTS", "0") == "1":
            print("  [INPUTS RECONSTRUCTED]")
            preview(inputs, indent=2)

        parent_tuple = _pick_parent_by_structure_and_description(
            scf_records=scf_recs,
            target_structure=struct,
            target_description=desc,
        )
        if not parent_tuple:
            print("  -> [ERROR] 親 SCF が見つからないか、data-file-schema.xml が見当たりません。")
            continue

        scf_remote, scf_calc, from_uuid = parent_tuple
        print(f"  -> [PARENT] {struct}: pick SCF PwCalculation<{scf_calc.pk}> from uuid={from_uuid}")
        print(f"     parent path = {scf_remote.get_remote_path()} (computer={scf_remote.computer.label})")

        # NSCF parent_folder
        nscf_pw = inputs.setdefault("nscf", {}).setdefault("pw", {})
        nscf_pw["parent_folder"] = scf_remote

        if proc_cls is Wannier90OptimizeWorkChain:
            ensure_bands_kpoints_from_reference_bands(inputs)

        # Code 差し替え
        rebind_codes(inputs, CODE_MAP_HEMULEN)

        # 旧計算から引き継がれた parallelization を全削除
        remove_qe_parallelization_inputs_everywhere(inputs)

        # seekpath 無効 + 元構造固定
        disable_seekpath_and_clean_artifacts(inputs)
        force_nscf_structure_from_original(inputs)

        # CWF 有効化
        enable_cwf_mode(inputs)

        # NSCF
        if "nscf" in inputs and "pw" in inputs["nscf"]:
            ensure_resources_under(
                inputs["nscf"]["pw"],
                mpi=FIXED_MPI,
                wallclock_s=DEFAULT_WALLCLOCK,
                force=True,
            )
            inherit_parent_prefix_from_scf(inputs, scf_calc)
            force_empty_settings(inputs["nscf"]["pw"], "NSCF")

        # PROJWFC
        if "projwfc" in inputs and "projwfc" in inputs["projwfc"]:
            ensure_resources_under(
                inputs["projwfc"]["projwfc"],
                mpi=FIXED_MPI,
                wallclock_s=DEFAULT_WALLCLOCK,
                force=True,
            )
            force_empty_settings(inputs["projwfc"]["projwfc"], "PROJWFC")

        # PW2WANNIER90
        if "pw2wannier90" in inputs and "pw2wannier90" in inputs["pw2wannier90"]:
            ensure_resources_under(
                inputs["pw2wannier90"]["pw2wannier90"],
                mpi=FIXED_MPI,
                wallclock_s=DEFAULT_WALLCLOCK,
                force=True,
            )
            sanitize_pw2wannier90_parameters(inputs)
            force_empty_settings(inputs["pw2wannier90"]["pw2wannier90"], "PW2WAN")

        # WANNIER90
        if "wannier90" in inputs and "wannier90" in inputs["wannier90"]:
            ensure_resources_under(
                inputs["wannier90"]["wannier90"],
                mpi=W90_MPI,
                wallclock_s=DEFAULT_WALLCLOCK,
                force=True,
            )
            sanitize_wannier90_settings(inputs)
            sanitize_wannier90_parameters(inputs)
            ensure_required_w90_kpoints(inputs)
            ensure_bands_plot_true(inputs)
            ensure_wannier90_serial_by_resources_only(inputs)

        # 念のため outdir 一掃
        _strip_outdir_anywhere(inputs)

        debug_print_parallel_state(inputs)

        if DRY_RUN:
            print("  -> DRY_RUN=1: execution skipped.")
            continue

        try:
            if USE_SUBMIT:
                future = submit(proc_cls, **inputs)
                print(f"  -> SUBMITTED: pk={future.pk}, uuid={future.uuid}")
                submitted_rows.append((sec, struct, str(future.uuid), future.process_label, desc))
                sections_written[sec].append((struct, str(future.uuid), future.process_label, desc))
            else:
                print("  -> RUN starting (this will block until finished)...")
                _ = run(proc_cls, **inputs)
                print("  -> RUN finished.")
        except Exception as e:
            print(f"  -> [ERROR] execution failed: {e}")

    if submitted_rows:
        out_path = os.path.splitext(wannier_list_path)[0] + "_hemulen.txt"

        structure_w = max(len("structure"), max((len(struct) for _, struct, _, _, _ in submitted_rows), default=0))
        uuid_w = max(len("uuid"), max((len(u) for _, _, u, _, _ in submitted_rows), default=0))
        plabel_w = max(len("Process label"), max((len(pl) for _, _, _, pl, _ in submitted_rows), default=0))

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                f"{'structure':<{structure_w}}  "
                f"{'uuid':<{uuid_w}}  "
                f"{'Process label':<{plabel_w}}  "
                f"description\n"
            )
            f.write(
                f"{'-' * structure_w}  " f"{'-' * uuid_w}  " f"{'-' * plabel_w}  " f"{'-' * len('description')}\n\n"
            )

            for sec, rows in sections_written.items():
                f.write(f"{sec}\n")
                for struct, u, pl, desc in rows:
                    f.write(f"  {struct:<{structure_w}}  {u:<{uuid_w}}  {pl:<{plabel_w}}")
                    if desc:
                        f.write(f"  {desc}")
                    f.write("\n")
                f.write("\n")

        print(f"[OK] Wrote {len(submitted_rows)} rows across {len(sections_written)} sections to {out_path}")
    else:
        print("[OK] Wrote 0 rows (no submissions)")


# ==========================
# エントリポイント
# ==========================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: USE_SUBMIT=1 python launch_cw_atom_proj.py <wannier_list.txt> <scf_list.txt>",
            file=sys.stderr,
        )
        sys.exit(2)

    main(sys.argv[1], sys.argv[2])
