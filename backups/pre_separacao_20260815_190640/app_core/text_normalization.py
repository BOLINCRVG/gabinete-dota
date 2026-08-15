"""UI text normalisation: mojibake repair, key canonicalisation, and Streamlit text-rendering patch."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, List

import pandas as pd
import streamlit as st

_MOJIBAKE_MARKERS = ("\ufffd", "\u00c3", "\u00c2", "\u00e2")


def _looks_like_mojibake(s: str) -> bool:
    if any(m in s for m in _MOJIBAKE_MARKERS):
        return True
    if "??" in s:
        return True
    if bool(re.search(r"[A-Za-z\u00c0-\u00ff]\?[A-Za-z\u00c0-\u00ff]", s)):
        return True
    return False


def _fix_mojibake_text(s: Any) -> Any:
    if not isinstance(s, str):
        return s

    def _score(txt: str) -> tuple[int, int]:
        bad = 0
        bad += txt.count("\ufffd")
        bad += txt.count("\u00c3") + txt.count("\u00c2") + txt.count("\u00e2")
        bad += len(re.findall(r"[A-Za-z\u00c0-\u00ff]\?[A-Za-z\u00c0-\u00ff]", txt))
        bad += txt.count("??")
        return (bad, len(txt))

    def _replace_common(txt: str) -> str:
        rep = {
            "N\ufffdo": "N\u00e3o",
            "n\ufffdo": "n\u00e3o",
            "N?o": "N\u00e3o",
            "n?o": "n\u00e3o",
            "S\ufffdrie": "S\u00e9rie",
            "s\ufffdrie": "s\u00e9rie",
            "S?rie": "S\u00e9rie",
            "s?rie": "s\u00e9rie",
            "M\ufffddia": "M\u00e9dia",
            "m\ufffddia": "m\u00e9dia",
            "M?dia": "M\u00e9dia",
            "m?dia": "m\u00e9dia",
            "Hist\ufffdrico": "Hist\u00f3rico",
            "hist\ufffdrico": "hist\u00f3rico",
            "Hist?rico": "Hist\u00f3rico",
            "hist?rico": "hist\u00f3rico",
            "Configura\ufffd\ufffdes": "Configura\u00e7\u00f5es",
            "Calibra\ufffd\ufffdes": "Calibra\u00e7\u00f5es",
            "Par\ufffdmetros": "Par\u00e2metros",
            "par\ufffdmetros": "par\u00e2metros",
            "Configura??es": "Configura\u00e7\u00f5es",
            "Calibra??es": "Calibra\u00e7\u00f5es",
            "Par?metros": "Par\u00e2metros",
            "par?metros": "par\u00e2metros",
            "Drag\ufffdes": "Drag\u00f5es",
            "Bar\ufffdes": "Bar\u00f5es",
            "Campe\ufffdes": "Campe\u00f5es",
            "Drag?es": "Drag\u00f5es",
            "Bar?es": "Bar\u00f5es",
            "Campe?es": "Campe\u00f5es",
            "Vit\ufffdria": "Vit\u00f3ria",
            "Vit?ria": "Vit\u00f3ria",
            "\ufffdltimos": "\u00daltimos",
            "?ltimos": "\u00daltimos",
            "v\ufffdlido": "v\u00e1lido",
            "inv\ufffdlido": "inv\u00e1lido",
            "r\ufffdpido": "r\u00e1pido",
            "R\ufffdpido": "R\u00e1pido",
            "sess\ufffdo": "sess\u00e3o",
            "a\ufffd\ufffdo": "a\u00e7\u00e3o",
            "for\ufffda": "for\u00e7a",
            "diferen\ufffda": "diferen\u00e7a",
            "M\ufffdtrica": "M\u00e9trica",
            "v?lido": "v\u00e1lido",
            "inv?lido": "inv\u00e1lido",
            "r?pido": "r\u00e1pido",
            "R?pido": "R\u00e1pido",
            "sess?o": "sess\u00e3o",
            "a??o": "a\u00e7\u00e3o",
            "A??o": "A\u00e7\u00e3o",
            "A?o": "A\u00e7\u00e3o",
            "for?a": "for\u00e7a",
            "diferen?a": "diferen\u00e7a",
            "M?trica": "M\u00e9trica",
            "Di?rio": "Di\u00e1rio",
            "di?rio": "di\u00e1rio",
            "Cabe?alho": "Cabe\u00e7alho",
            "cabe?alho": "cabe\u00e7alho",
            "obrigat?rias": "obrigat\u00f3rias",
            "obrigat?ria": "obrigat\u00f3ria",
            "Diagn?stico": "Diagn\u00f3stico",
            "diagn?stico": "diagn\u00f3stico",
            "Vis\ufffdo": "Vis\u00e3o",
            "Consist\ufffdncia": "Consist\u00eancia",
            "An\ufffdlise": "An\u00e1lise",
            "Navega\ufffd\ufffdo": "Navega\u00e7\u00e3o",
            "Vis?o": "Vis\u00e3o",
            "Consist?ncia": "Consist\u00eancia",
            "An?lise": "An\u00e1lise",
            "Navega??o": "Navega\u00e7\u00e3o",
            "opera\ufffd\ufffdo": "opera\u00e7\u00e3o",
            "promo\ufffd\ufffdo": "promo\u00e7\u00e3o",
            "regress\ufffdo": "regress\u00e3o",
            "integra\ufffd\ufffdo": "integra\u00e7\u00e3o",
            "manuten\ufffd\ufffdo": "manuten\u00e7\u00e3o",
            "vers\ufffdo": "vers\u00e3o",
            "recalibra\ufffd\ufffdo": "recalibra\u00e7\u00e3o",
            "calibra\ufffd\ufffdo": "calibra\u00e7\u00e3o",
            "compara\ufffd\ufffdo": "compara\u00e7\u00e3o",
            "pre\ufffdo": "pre\u00e7o",
            "opera??o": "opera\u00e7\u00e3o",
            "promo??o": "promo\u00e7\u00e3o",
            "regress?o": "regress\u00e3o",
            "integra??o": "integra\u00e7\u00e3o",
            "manuten??o": "manuten\u00e7\u00e3o",
            "vers?o": "vers\u00e3o",
            "recalibra??o": "recalibra\u00e7\u00e3o",
            "calibra??o": "calibra\u00e7\u00e3o",
            "compara??o": "compara\u00e7\u00e3o",
            "pre?o": "pre\u00e7o",
            "valida??o": "valida\u00e7\u00e3o",
            "valida??es": "valida\u00e7\u00f5es",
            "prepara??o": "prepara\u00e7\u00e3o",
            "governan?a": "governan\u00e7a",
            "su?te": "su\u00edte",
            "s?o": "s\u00e3o",
            "pr?-": "pr\u00e9-",
            "P?gina": "P\u00e1gina",
            "Fluxo R\ufffdpido": "Fluxo R\u00e1pido",
            "Fluxo R\u00e1pido": "Fluxo R\u00e1pido",
            "?ncoras": "\u00e2ncoras",
            "?ncora": "\u00e2ncora",
            "?ntegro": "\u00edntegro",
            "?ntegra": "\u00edntegra",
            "?ltima": "\u00faltima",
            "?ltimo": "\u00faltimo",
            "?nico": "\u00fanico",
            "?nica": "\u00fanica",
            "?rf?o": "\u00f3rf\u00e3o",
            "?til": "\u00fatil",
            "?xito": "\u00eaxito",
            "T?cnico": "T\u00e9cnico",
            "Pr?-entrega": "Pr\u00e9-entrega",
            "Pr?": "Pr\u00f3",
            "ATEN??O": "ATEN\u00c7\u00c3O",
            "Aten??o": "Aten\u00e7\u00e3o",
            "Par?metro": "Par\u00e2metro",
            "Padr?o": "Padr\u00e3o",
            "padr?o": "padr\u00e3o",
            "Avan?ado": "Avan\u00e7ado",
            "For?a": "For\u00e7a",
            "M?x": "M\u00e1x",
            "m?x": "m\u00e1x",
            "m?ximo": "m\u00e1ximo",
            "m?xima": "m\u00e1xima",
            "m?nima": "m\u00ednima",
            "n?mero": "n\u00famero",
            "n?veis": "n\u00edveis",
            "num?rico": "num\u00e9rico",
            "confian?a": "confian\u00e7a",
            "Confian?a": "Confian\u00e7a",
            "consist?ncia": "consist\u00eancia",
            "Diverg?ncia": "Diverg\u00eancia",
            "diverg?ncia": "diverg\u00eancia",
            "coer?ncia": "coer\u00eancia",
            "c?lculo": "c\u00e1lculo",
            "calend?rio": "calend\u00e1rio",
            "campe?es": "campe\u00f5es",
            "campe?o": "campe\u00e3o",
            "Campe?o": "Campe\u00e3o",
            "advers?rio": "advers\u00e1rio",
            "espec?ficas": "espec\u00edficas",
            "espec?fico": "espec\u00edfico",
            "expl?cito": "expl\u00edcito",
            "leg?vel": "leg\u00edvel",
            "refer?ncia": "refer\u00eancia",
            "relat?rio": "relat\u00f3rio",
            "Relat?rio": "Relat\u00f3rio",
            "r?tulo": "r\u00f3tulo",
            "saud?vel": "saud\u00e1vel",
            "sa?da": "sa\u00edda",
            "sugest?es": "sugest\u00f5es",
            "v?rgula": "v\u00edrgula",
            "voc?": "voc\u00ea",
            "Voc?": "Voc\u00ea",
            "h?": "h\u00e1",
            "s?": "s\u00f3",
            "est?": "est\u00e1",
            "j?": "j\u00e1",
            "tamb?m": "tamb\u00e9m",
            "m?s": "m\u00eas",
            "ocorr?ncia": "ocorr\u00eancia",
            "Ocorr?ncias": "Ocorr\u00eancias",
            "Hor?rio": "Hor\u00e1rio",
            "Presen?a": "Presen\u00e7a",
            "presen?a": "presen\u00e7a",
            "persist?ncia": "persist\u00eancia",
            "audit?vel": "audit\u00e1vel",
            "heur?stico": "heur\u00edstico",
            "altera??es": "altera\u00e7\u00f5es",
            "execu??o": "execu\u00e7\u00e3o",
            "migra??es": "migra\u00e7\u00f5es",
            "oscila??es": "oscila\u00e7\u00f5es",
            "produ??o": "produ\u00e7\u00e3o",
            "Promo??o": "Promo\u00e7\u00e3o",
            "Manuten??o": "Manuten\u00e7\u00e3o",
            "calibra??es/fus?es": "calibra\u00e7\u00f5es/fus\u00f5es",
            "Auto-corre??o": "Auto-corre\u00e7\u00e3o",
            "auto-corre??o": "auto-corre\u00e7\u00e3o",
            "conclu?da": "conclu\u00edda",
            "Conclu?do": "Conclu\u00eddo",
            "conclu?do": "conclu\u00eddo",
            "pend?ncia": "pend\u00eancia",
            "pend?ncias": "pend\u00eancias",
            "cr?tica": "cr\u00edtica",
            "cr?ticas": "cr\u00edticas",
            "cr?ticos": "cr\u00edticos",
            "h?brido": "h\u00edbrido",
            "din?mico": "din\u00e2mico",
            "hist?rica": "hist\u00f3rica",
            "in?cio": "in\u00edcio",
            "indispon?vel": "indispon\u00edvel",
            "come?am": "come\u00e7am",
            "come?ar": "come\u00e7ar",
            "come?o": "come\u00e7o",
            "espa?o": "espa\u00e7o",
            "gen?ricos": "gen\u00e9ricos",
            "hier?rquica": "hier\u00e1rquica",
            "cont?nua": "cont\u00ednua",
            "secund?rias": "secund\u00e1rias",
            "sem?ntica": "sem\u00e2ntica",
            "toler?ncia": "toler\u00e2ncia",
            "uni?o": "uni\u00e3o",
            "usu?rio": "usu\u00e1rio",
            "pain?is": "pain\u00e9is",
            "Pain?is": "Pain\u00e9is",
            "vis?vel": "vis\u00edvel",
            "di?ria": "di\u00e1ria",
            "autom?tica": "autom\u00e1tica",
            "autom?tico": "autom\u00e1tico",
            "autom?ticos": "autom\u00e1ticos",
            "autom?ticas": "autom\u00e1ticas",
            "drift/pend?ncia": "drift/pend\u00eancia",
            "rec?ncia": "rec\u00eancia",
            "rec?ncia/estado": "rec\u00eancia/estado",
            "est?vel": "est\u00e1vel",
            "est?veis": "est\u00e1veis",
            "ap?s": "ap\u00f3s",
            "constru?do": "constru\u00eddo",
            "dispon?veis": "dispon\u00edveis",
            "auditoria/reprodu??o": "auditoria/reprodu\u00e7\u00e3o",
            "T?cnico/Par": "T\u00e9cnico/Par",
            "?? Diverg?ncia": "Diverg\u00eancia",
            "(?=": "(\u0394=",
        }
        out = txt
        for k, v in rep.items():
            out = out.replace(k, v)
        return out

    if not _looks_like_mojibake(s):
        return s

    candidates = [s, _replace_common(s)]
    for enc in ("latin1", "cp1252"):
        try:
            dec = s.encode(enc).decode("utf-8")
            candidates.append(dec)
            candidates.append(_replace_common(dec))
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    best = min(candidates, key=_score)
    return best if _score(best) < _score(s) else _replace_common(s)


def _normalize_text_key(s: Any) -> str:
    txt = _fix_mojibake_text(str(s or ""))
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[^A-Za-z0-9]+", "", txt).casefold()
    return txt


def _canonicalize_tab_label(label: Any, canonical_tabs: List[str]) -> str:
    fixed = _fix_mojibake_text(str(label or "")).strip()
    if not fixed:
        return fixed

    fixed_key = _normalize_text_key(fixed)
    best_label = ""
    best_ratio = 0.0
    for can in canonical_tabs:
        can_key = _normalize_text_key(can)
        if fixed_key == can_key:
            return can
        ratio = difflib.SequenceMatcher(None, fixed_key, can_key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_label = can
    if best_label and best_ratio >= 0.90:
        return best_label
    return fixed


def _canonicalize_tab_labels(tabs: List[Any], canonical_tabs: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for tab in tabs:
        can = _canonicalize_tab_label(tab, canonical_tabs)
        key = _normalize_text_key(can)
        if key in seen:
            continue
        seen.add(key)
        out.append(can)
    return out


def _patch_streamlit_text_rendering() -> None:
    if getattr(st, "_gabinete_text_patch", False):
        return

    def _fix_obj(x):
        if isinstance(x, str):
            return _fix_mojibake_text(x)
        if isinstance(x, list):
            return [_fix_obj(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_fix_obj(v) for v in x)
        if isinstance(x, dict):
            return {k: _fix_obj(v) for k, v in x.items()}
        return x

    def _wrap_first_arg(fn):
        def _inner(*args, **kwargs):
            if args:
                args = tuple(_fix_obj(a) for a in args)
            if kwargs:
                kwargs = {k: _fix_obj(v) for k, v in kwargs.items()}
            return fn(*args, **kwargs)
        return _inner

    def _wrap_tabs(fn):
        def _inner(tabs, *args, **kwargs):
            fixed_tabs = [_fix_mojibake_text(x) if isinstance(x, str) else x for x in tabs]
            return fn(fixed_tabs, *args, **kwargs)
        return _inner

    def _wrap_expander(fn):
        def _inner(label, *args, **kwargs):
            # UX rule: todos os expanders começam minimizados.
            kwargs["expanded"] = False
            return fn(_fix_mojibake_text(label) if isinstance(label, str) else label, *args, **kwargs)
        return _inner

    def _fix_dataframe_obj(obj):
        if isinstance(obj, pd.DataFrame):
            try:
                d = obj.copy()
                d.columns = [(_fix_mojibake_text(c) if isinstance(c, str) else c) for c in d.columns]
                try:
                    d.index = [(_fix_mojibake_text(i) if isinstance(i, str) else i) for i in d.index]
                except (TypeError, ValueError):
                    pass
                for c in d.columns:
                    try:
                        if pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
                            d[c] = d[c].map(lambda v: _fix_mojibake_text(v) if isinstance(v, str) else v)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                return d
            except (AttributeError, KeyError, TypeError, ValueError):
                return obj
        return obj

    def _wrap_dataframe(fn):
        def _inner(data=None, *args, **kwargs):
            data2 = _fix_dataframe_obj(data)
            return fn(data2, *args, **kwargs)
        return _inner

    def _as_list(options: Any) -> List[Any]:
        try:
            if options is None:
                return []
            if isinstance(options, list):
                return options
            if isinstance(options, tuple):
                return list(options)
            if isinstance(options, pd.Series):
                return options.tolist()
            return list(options)
        except TypeError:
            return []

    def _contains_opt(opts: List[Any], value: Any) -> bool:
        for o in opts:
            try:
                if o == value:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _safe_default_from_options(opts: List[Any], index: int | None = None) -> Any:
        if not opts:
            return None
        if index is None:
            return opts[0]
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = 0
        if i < 0 or i >= len(opts):
            i = 0
        return opts[i]

    def _wrap_selector_single(fn):
        # radio/selectbox: sanitize stale session value before widget instantiation.
        def _inner(*args, **kwargs):
            lbl = _fix_obj(args[0]) if args else _fix_obj(kwargs.get("label"))
            opts = []
            if len(args) >= 2:
                opts = _as_list(_fix_obj(args[1]))
            elif "options" in kwargs:
                opts = _as_list(_fix_obj(kwargs.get("options")))
            key = kwargs.get("key")
            idx = kwargs.get("index")
            if key is not None:
                cur = st.session_state.get(key, None)
                if opts:
                    if not _contains_opt(opts, cur):
                        st.session_state[key] = _safe_default_from_options(opts, idx)
                else:
                    # Empty dynamic selector: clear stale values so Streamlit does not try to serialize
                    # a session value that is no longer present in options.
                    if cur not in (None, ""):
                        st.session_state[key] = None
            if opts:
                try:
                    i = kwargs.get("index", None)
                    if i is not None:
                        ii = int(i)
                        if ii < 0 or ii >= len(opts):
                            kwargs["index"] = 0
                except (TypeError, ValueError):
                    kwargs["index"] = 0
            else:
                kwargs.pop("index", None)
            # Rebuild args/kwargs with fixed text/objects.
            if args:
                args = tuple(_fix_obj(a) for a in args)
            if kwargs:
                kwargs = {k: _fix_obj(v) for k, v in kwargs.items()}
            if args:
                return fn(*args, **kwargs)
            return fn(lbl, options=opts, **kwargs)
        return _inner

    def _wrap_selector_multi(fn):
        # multiselect: keep only valid selections from stale session list.
        def _inner(*args, **kwargs):
            lbl = _fix_obj(args[0]) if args else _fix_obj(kwargs.get("label"))
            opts = []
            if len(args) >= 2:
                opts = _as_list(_fix_obj(args[1]))
            elif "options" in kwargs:
                opts = _as_list(_fix_obj(kwargs.get("options")))
            key = kwargs.get("key")
            if key is not None:
                cur = st.session_state.get(key, None)
                if not isinstance(cur, (list, tuple, set)):
                    cur = []
                if opts:
                    cur_list = [x for x in list(cur) if _contains_opt(opts, x)]
                else:
                    cur_list = []
                st.session_state[key] = cur_list
            if "default" in kwargs:
                dflt = kwargs.get("default")
                if not isinstance(dflt, (list, tuple, set)):
                    dflt = [] if dflt is None else [dflt]
                if opts:
                    kwargs["default"] = [x for x in list(dflt) if _contains_opt(opts, x)]
                else:
                    kwargs["default"] = []
            if args:
                args = tuple(_fix_obj(a) for a in args)
            if kwargs:
                kwargs = {k: _fix_obj(v) for k, v in kwargs.items()}
            if args:
                return fn(*args, **kwargs)
            return fn(lbl, options=opts, **kwargs)
        return _inner

    # Keep runtime text correction focused on external inputs only.
    for name in [
        "text_input",
        "text_area",
        "file_uploader",
    ]:
        if hasattr(st, name):
            setattr(st, name, _wrap_first_arg(getattr(st, name)))
    if hasattr(st, "tabs"):
        st.tabs = _wrap_tabs(st.tabs)
    if hasattr(st, "expander"):
        st.expander = _wrap_expander(st.expander)
    if hasattr(st, "dataframe"):
        st.dataframe = _wrap_dataframe(st.dataframe)
    if hasattr(st, "table"):
        st.table = _wrap_dataframe(st.table)
    # Extra hardening for stale session_state in dynamic selectors.
    if hasattr(st, "selectbox"):
        st.selectbox = _wrap_selector_single(st.selectbox)
    if hasattr(st, "radio"):
        st.radio = _wrap_selector_single(st.radio)
    if hasattr(st, "multiselect"):
        st.multiselect = _wrap_selector_multi(st.multiselect)

    st._gabinete_text_patch = True
