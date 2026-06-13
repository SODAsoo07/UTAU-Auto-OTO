from __future__ import annotations

from ui.layout_mixin import DEVELOPER_ONLY_TABS, TAB_ORDER, LayoutMixin
from ui import tab_builders_mixin
from ui.tab_builders_mixin import PARAM_GROUPS, TabBuildersMixin


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _FakeTabview:
    def __init__(self, names, current):
        self._tab_dict = {name: object() for name in names}
        self._current = current

    def get(self):
        return self._current

    def set(self, name):
        if name not in self._tab_dict:
            raise ValueError(name)
        self._current = name

    def insert(self, index, name):
        items = list(self._tab_dict.items())
        items.insert(index, (name, object()))
        self._tab_dict = dict(items)

    def delete(self, name):
        del self._tab_dict[name]


class _LayoutHarness(LayoutMixin):
    pass


class _ParamHarness(TabBuildersMixin):
    pass


def test_basic_view_hides_developer_tabs_and_clears_destroyed_widget_refs():
    app = _LayoutHarness()
    app.developer_mode_enabled_var = _FakeVar(False)
    app.tabview = _FakeTabview(TAB_ORDER, current="상세 로그")
    app._built_tabs = {"파이프라인", "파라미터 조정", "상세 로그"}
    app._tab_build_pending = {"상세 로그"}
    app._lazy_prewarm_queue = ["로그", "상세 로그", "크레딧"]
    app.detail_log_text = object()

    app._sync_secondary_tab_visibility(False)

    assert tuple(app.tabview._tab_dict) == ("파이프라인", "고급 설정", "로그", "크레딧")
    assert app.tabview.get() == "파이프라인"
    assert not (app._built_tabs & DEVELOPER_ONLY_TABS)
    assert "상세 로그" not in app._tab_build_pending
    assert app._lazy_prewarm_queue == ["로그", "크레딧"]
    assert app.detail_log_text is None


def test_advanced_view_restores_developer_tabs_in_stable_order():
    app = _LayoutHarness()
    app.developer_mode_enabled_var = _FakeVar(True)
    app.tabview = _FakeTabview(("파이프라인", "고급 설정", "로그", "크레딧"), current="파이프라인")

    app._sync_secondary_tab_visibility(True)

    assert tuple(app.tabview._tab_dict) == TAB_ORDER


def test_param_vars_exist_without_building_parameter_tab(monkeypatch):
    monkeypatch.setattr(tab_builders_mixin.ctk, "DoubleVar", lambda value: _FakeVar(value))
    app = _ParamHarness()

    app._ensure_param_vars()
    expected_keys = {key for _group_name, params in PARAM_GROUPS for key, *_rest in params}
    assert set(app.param_vars) == expected_keys

    app.param_vars["VC_PRE_OFFSET"].set(17)
    app._ensure_param_vars()
    assert app.param_vars["VC_PRE_OFFSET"].get() == 17
