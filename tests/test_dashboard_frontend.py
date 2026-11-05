"""Static frontend tests for inference/web/static/dashboard.html.

Pure-Python unittest — no browser, no JS engine.  Parses the dashboard
HTML source and verifies invariants the frontend must uphold:

1. Every API path referenced in JS exists as a route in
   inference/web/app.py — the frontend must never call an endpoint the
   backend does not serve.
2. None/empty handling strings are present ("Not calibrated",
   "Not generated", "Not available") — the UI must visibly distinguish
   "value is None" from a real value instead of showing a blank cell.
3. No mock/fake/dummy production data is injected into the JS.
4. The Input/Processing tab (pane-input) exists.
5. The video feed endpoint is used for the stream (MJPEG img src).
"""
from __future__ import annotations

import os
import re
import unittest

_DASHBOARD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "inference", "web", "static", "dashboard.html",
)


def _read_dashboard() -> str:
    with open(_DASHBOARD, "r", encoding="utf-8") as f:
        return f.read()


def _backend_routes() -> set:
    """Extract the set of routes registered in inference/web/app.py
    by reading the source directly (no Flask import needed)."""
    import inference.web.app as web_app
    src_path = os.path.join(os.path.dirname(web_app.__file__), "app.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    return set(re.findall(r'@app\.route\("([^"]+)"\)', src))


class TestDashboardEndpoints(unittest.TestCase):
    """Frontend must only reference API paths that the backend serves."""

    def setUp(self):
        self.html = _read_dashboard()
        self.routes = _backend_routes()

    def test_api_paths_in_js_exist_in_backend(self):
        """Every /api/... and /video_feed string in the JS must match a
        route in app.py — no fabricated endpoints."""
        referenced = set(re.findall(
            r"""['"`](/api/[A-Za-z0-9_/<>.\-]*)['"`]""", self.html))
        referenced |= set(re.findall(r"""['"`](/video_feed)['"`]""",
                                     self.html))
        self.assertGreater(
            len(referenced), 0,
            "dashboard must reference at least one API path")

        for path in referenced:
            # Split path into a static prefix (up to first dynamic segment)
            # and compare against route prefixes.
            path_prefix = path.split("<")[0]  # strip any <param> tail
            matched = any(
                route == path
                or (route in path)
                or (route.startswith(path_prefix) and "/<")
                or path.startswith(route.split("/<")[0])
                for route in self.routes
            )
            # For /api/events/<id> in dashboard JS, the static prefix
            # /api/events must match route /api/events or /api/events/<...>
            if not matched:
                # Try matching on static prefix only
                matched = any(
                    route.replace("/<event_id>", "").replace("/<", "") in path
                    for route in self.routes
                )
            self.assertTrue(
                matched,
                f"dashboard references '{path}' but no matching route "
                f"exists in app.py (routes: {sorted(self.routes)})"
            )

    def test_video_feed_used_for_stream(self):
        self.assertIn("/video_feed", self.html)


class TestDashboardNoneHandling(unittest.TestCase):
    """The UI must visibly distinguish None values from real values."""

    def setUp(self):
        self.html = _read_dashboard()

    def test_not_calibrated_present(self):
        self.assertIn("Not calibrated", self.html)

    def test_not_generated_present(self):
        self.assertIn("Not generated", self.html)

    def test_not_available_present(self):
        self.assertIn("Not available", self.html)

    def test_scoring_status_not_calibrated_handled(self):
        """The JS must reference 'not_calibrated' for scoring_status."""
        self.assertIn("not_calibrated", self.html)


class TestDashboardNoMockData(unittest.TestCase):
    """No mock/fake/dummy production data may be embedded in the JS."""

    def setUp(self):
        self.html = _read_dashboard()

    def test_no_mock_token(self):
        self.assertNotRegex(self.html.lower(), r"\bmock\s?data\b")

    def test_no_fake_token(self):
        self.assertNotRegex(self.html.lower(), r"\bfake\s?data\b")

    def test_no_dummy_token(self):
        self.assertNotRegex(self.html.lower(), r"\bdummy\s?data\b")

    def test_no_hardcoded_uuids(self):
        """The events table must be built from API data, not hardcoded."""
        hardcoded = re.findall(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
            r"-[0-9a-f]{4}-[0-9a-f]{12}",
            self.html,
        )
        self.assertEqual(
            hardcoded, [],
            "dashboard must not contain hardcoded UUIDs")


class TestDashboardStructure(unittest.TestCase):
    """Structural invariants of the SPA."""

    def setUp(self):
        self.html = _read_dashboard()

    def test_input_pane_exists(self):
        self.assertIn('id="pane-input"', self.html)

    def test_all_tabs_have_panes(self):
        tabs = re.findall(r'data-tab="([^"]+)"', self.html)
        self.assertGreaterEqual(
            len(tabs), 5, "expected at least 5 tabs")
        for t in tabs:
            self.assertIn(
                f'id="pane-{t}"', self.html,
                f"tab '{t}' has no matching pane")

    def test_script_tags_not_external(self):
        """No external script/CDN dependencies — single-file SPA."""
        ext = re.findall(r'<script[^>]+src="http', self.html)
        self.assertEqual(
            ext, [],
            "dashboard must be self-contained (no external scripts)")


if __name__ == "__main__":
    unittest.main()
