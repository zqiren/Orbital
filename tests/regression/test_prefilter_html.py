# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: HTML pre-filter strips noise from browser fetch results.

The pre-filter uses trafilatura to extract article content, stripping
navigation, footer, scripts, and CSS from raw HTML.
"""

import pytest

from agent_os.agent.tool_result_filters import dispatch_prefilter, _prefilter_html


# Sample HTML page with nav, footer, scripts, and article content
SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Test Article</title>
    <script src="analytics.js"></script>
    <style>
        body { font-family: sans-serif; }
        .nav { background: #333; color: white; }
        .footer { background: #eee; padding: 20px; }
    </style>
</head>
<body>
    <nav class="nav">
        <a href="/">Home</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
        <a href="/blog">Blog</a>
        <a href="/products">Products</a>
    </nav>

    <div class="sidebar">
        <h3>Related Articles</h3>
        <ul>
            <li><a href="/article1">Article 1</a></li>
            <li><a href="/article2">Article 2</a></li>
            <li><a href="/article3">Article 3</a></li>
        </ul>
        <div class="ad-banner">Advertisement placeholder text here</div>
    </div>

    <article>
        <h1>Understanding Machine Learning Fundamentals</h1>
        <p>Machine learning is a subset of artificial intelligence that focuses on
        building systems that learn from data. Rather than being explicitly programmed,
        these systems improve their performance on tasks through experience.</p>

        <h2>Types of Machine Learning</h2>
        <p>There are three main types of machine learning: supervised learning,
        unsupervised learning, and reinforcement learning. Each approach has its
        own strengths and is suited to different types of problems.</p>

        <h3>Supervised Learning</h3>
        <p>In supervised learning, the model is trained on labeled data. The algorithm
        learns to map input features to known output labels. Common examples include
        classification tasks like spam detection and regression tasks like price
        prediction.</p>

        <h3>Unsupervised Learning</h3>
        <p>Unsupervised learning works with unlabeled data. The algorithm tries to
        find patterns and structure in the data without prior knowledge of what the
        output should be. Clustering and dimensionality reduction are common
        unsupervised techniques.</p>

        <h3>Reinforcement Learning</h3>
        <p>Reinforcement learning involves an agent that learns to make decisions
        by interacting with an environment. The agent receives rewards or penalties
        for its actions and learns to maximize cumulative reward over time.</p>
    </article>

    <footer class="footer">
        <p>Copyright 2024 Example Corp. All rights reserved.</p>
        <p>Terms of Service | Privacy Policy | Cookie Policy</p>
        <div class="social-links">
            <a href="https://twitter.com/example">Twitter</a>
            <a href="https://facebook.com/example">Facebook</a>
        </div>
    </footer>

    <script>
        // Analytics tracking
        var ga = document.createElement('script');
        ga.src = 'https://analytics.example.com/track.js';
        document.head.appendChild(ga);

        // Cookie consent
        function showCookieBanner() {
            var banner = document.getElementById('cookie-banner');
            banner.style.display = 'block';
        }
    </script>
</body>
</html>"""


class TestPrefilterHTML:
    """HTML pre-filter extracts article content and removes noise."""

    def test_output_smaller_than_input(self):
        """Filtered output is significantly smaller than raw HTML."""
        filtered = _prefilter_html(SAMPLE_HTML)
        assert len(filtered) < len(SAMPLE_HTML) * 0.50, (
            f"Expected >50% reduction, got {len(filtered)}/{len(SAMPLE_HTML)} "
            f"= {len(filtered)/len(SAMPLE_HTML):.1%}"
        )

    def test_article_text_preserved(self):
        """Main article content is preserved in the filtered output."""
        filtered = _prefilter_html(SAMPLE_HTML)
        assert "Machine Learning Fundamentals" in filtered
        assert "supervised learning" in filtered.lower()
        assert "reinforcement learning" in filtered.lower()

    def test_non_html_returned_unchanged(self):
        """Non-HTML input is returned as-is."""
        plain_text = "This is just plain text, not HTML at all."
        filtered = _prefilter_html(plain_text)
        assert filtered == plain_text

    def test_malformed_html_no_crash(self):
        """Malformed HTML should not crash, returns input unchanged."""
        malformed = "<div><p>Unclosed tags <span>everywhere"
        result = _prefilter_html(malformed)
        # Should return something without crashing
        assert isinstance(result, str)

    def test_dispatch_routes_browser_fetch(self):
        """dispatch_prefilter routes browser fetch action to HTML filter."""
        filtered = dispatch_prefilter(
            "browser",
            {"action": "fetch", "url": "https://example.com"},
            SAMPLE_HTML,
        )
        # Should be filtered (smaller than original)
        assert len(filtered) < len(SAMPLE_HTML)

    def test_dispatch_does_not_filter_browser_snapshot(self):
        """dispatch_prefilter does NOT apply HTML filter to browser snapshot."""
        # Snapshot content is an accessibility tree, not HTML
        snapshot_content = (
            "[ref=e1] heading 'Main Page'\n"
            "[ref=e2] button 'Sign In'\n"
            "[ref=e3] link 'About Us'\n"
        ) * 200  # Make it large enough
        filtered = dispatch_prefilter(
            "browser",
            {"action": "snapshot"},
            snapshot_content,
        )
        # Should NOT be filtered by HTML filter (it's not HTML)
        assert filtered == snapshot_content

    def test_empty_content_unchanged(self):
        """Empty content returns empty."""
        assert dispatch_prefilter("browser", {"action": "fetch"}, "") == ""
