"""
Selector tests for cssselect backend
"""

from __future__ import annotations

import gc
import pickle
import weakref
from abc import ABC, abstractmethod
from typing import Any

import pytest
from cssselect import __version__ as CSSSELECT_VERSION_STRING
from cssselect.parser import SelectorSyntaxError
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
from packaging.version import Version

from parsel import Selector, css2xpath
from parsel.csstranslator import (
    GenericTranslator,
    HTMLTranslator,
    TranslatorProtocol,
    XPathExpr,
)

CSSSELECT_VERSION = Version(CSSSELECT_VERSION_STRING)

HTMLBODY = """
<html>
<body>
<div>
 <a id="name-anchor" name="foo"></a>
 <a id="tag-anchor" rel="tag" href="http://localhost/foo">link</a>
 <a id="nofollow-anchor" rel="nofollow" href="https://example.org"> link</a>
 <p id="paragraph">
   lorem ipsum text
   <b id="p-b">hi</b> <em id="p-em">there</em>
   <b id="p-b2">guy</b>
   <input type="checkbox" id="checkbox-unchecked" />
   <input type="checkbox" id="checkbox-disabled" disabled="" />
   <input type="text" id="text-checked" checked="checked" />
   <input type="hidden" />
   <input type="hidden" disabled="disabled" />
   <input type="checkbox" id="checkbox-checked" checked="checked" />
   <input type="checkbox" id="checkbox-disabled-checked"
          disabled="disabled" checked="checked" />
   <fieldset id="fieldset" disabled="disabled">
     <input type="checkbox" id="checkbox-fieldset-disabled" />
     <input type="hidden" />
   </fieldset>
 </p>
 <map name="dummymap">
   <area shape="circle" coords="200,250,25" href="foo.html" id="area-href" />
   <area shape="default" id="area-nohref" />
 </map>
</div>
<div class="cool-footer" id="foobar-div" foobar="ab bc cde">
    <span id="foobar-span">foo ter</span>
</div>
</body></html>
"""


class TestTranslatorBase(ABC):
    @property
    @abstractmethod
    def tr_cls(self) -> type[TranslatorProtocol]:
        raise NotImplementedError

    def c2x(self, css: str) -> str:
        return self.tr_cls().css_to_xpath(css)

    @pytest.mark.parametrize(
        ("css", "xpath"),
        [
            ("::attr(name)", "descendant-or-self::*/@name"),
            ("a::attr(href)", "descendant-or-self::a/@href"),
            (
                "a ::attr(img)",
                "descendant-or-self::a/descendant-or-self::*/@img",
            ),
            ("a > ::attr(class)", "descendant-or-self::a/*/@class"),
            (
                r"a::attr(\:foo)",
                "descendant-or-self::a/attribute::*[name() = ':foo']",
            ),
            (
                r"a::attr(foo\:bar)",
                "descendant-or-self::a/attribute::*[name() = 'foo:bar']",
            ),
        ],
    )
    def test_attr_function(self, css: str, xpath: str) -> None:
        assert self.c2x(css) == xpath, css

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::attr(12)", ExpressionError),
            ("::attr(34test)", ExpressionError),
            ("::attr(@href)", SelectorSyntaxError),
        ],
    )
    def test_attr_function_exception(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "xpath"),
        [
            ("::text", "descendant-or-self::text()"),
            ("p::text", "descendant-or-self::p/text()"),
            ("p ::text", "descendant-or-self::p/descendant-or-self::text()"),
            ("#id::text", "descendant-or-self::*[@id = 'id']/text()"),
            ("p#id::text", "descendant-or-self::p[@id = 'id']/text()"),
            (
                "p#id ::text",
                "descendant-or-self::p[@id = 'id']/descendant-or-self::text()",
            ),
            ("p#id > ::text", "descendant-or-self::p[@id = 'id']/*/text()"),
            (
                "p#id ~ ::text",
                "descendant-or-self::p[@id = 'id']/following-sibling::*/text()",
            ),
            ("a[href]::text", "descendant-or-self::a[@href]/text()"),
            (
                "a[href] ::text",
                "descendant-or-self::a[@href]/descendant-or-self::text()",
            ),
            (
                "p::text, a::text",
                "descendant-or-self::p/text() | descendant-or-self::a/text()",
            ),
        ],
    )
    def test_text_pseudo_element(self, css: str, xpath: str) -> None:
        assert self.c2x(css) == xpath, css

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::attribute(12)", ExpressionError),
            ("::text()", ExpressionError),
            ("::attr(@href)", SelectorSyntaxError),
        ],
    )
    def test_pseudo_function_exception(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::text-node", ExpressionError),
        ],
    )
    def test_unknown_pseudo_element(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            (":text", ExpressionError),
            (":attribute(name)", ExpressionError),
        ],
    )
    def test_unknown_pseudo_class(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)


class TestHTMLTranslator(TestTranslatorBase):
    tr_cls = HTMLTranslator


class TestGenericTranslator(TestTranslatorBase):
    tr_cls = GenericTranslator


@pytest.mark.parametrize("tr_cls", [GenericTranslator, HTMLTranslator])
class TestTranslatorCache:
    def test_cache(self, tr_cls: type[GenericTranslator | HTMLTranslator]) -> None:
        tr = tr_cls()
        assert tr.css_to_xpath("a::text") == tr.css_to_xpath("a::text")
        assert tr._cache.cache_info().hits == 1

    def test_garbage_collection(
        self, tr_cls: type[GenericTranslator | HTMLTranslator]
    ) -> None:
        tr = tr_cls()
        tr.css_to_xpath("a::text")
        ref = weakref.ref(tr)
        del tr
        gc.collect()
        assert ref() is None

    def test_pickle(self, tr_cls: type[GenericTranslator | HTMLTranslator]) -> None:
        tr = pickle.loads(pickle.dumps(tr_cls()))  # noqa: S301
        assert tr.css_to_xpath("a::text") == "descendant-or-self::a/text()"


def test_css2xpath() -> None:
    prefilter = (
        "contains(@class, 'some-class') and "
        if CSSSELECT_VERSION >= Version("1.6.0")
        else ""
    )
    expected_xpath = (
        f"descendant-or-self::*[@class and {prefilter}contains("
        "concat(' ', normalize-space(@class), ' '), ' some-class ')]"
    )
    assert css2xpath(".some-class") == expected_xpath


def test_xpathexpr_join_wrong_type() -> None:
    expr = XPathExpr(path="foo")
    with pytest.raises(ValueError, match="can only join expressions"):
        expr.join("/", OriginalXPathExpr(path="bar"))


class TestCSSSelector:
    sel = Selector(text=HTMLBODY)

    def x(self, *a: Any, **kw: Any) -> list[str]:
        return [v.strip() for v in self.sel.css(*a, **kw).extract() if v.strip()]

    def test_selector_simple(self) -> None:
        for x in self.sel.css("input"):
            assert isinstance(x, self.sel.__class__), x
        assert self.sel.css("input").extract() == [
            x.extract() for x in self.sel.css("input")
        ]

    def test_text_pseudo_element(self) -> None:
        assert self.x("#p-b2") == ['<b id="p-b2">guy</b>']
        assert self.x("#p-b2::text") == ["guy"]
        assert self.x("#p-b2 ::text") == ["guy"]
        assert self.x("#paragraph::text") == ["lorem ipsum text"]
        assert self.x("#paragraph ::text") == ["lorem ipsum text", "hi", "there", "guy"]
        assert self.x("p::text") == ["lorem ipsum text"]
        assert self.x("p ::text") == ["lorem ipsum text", "hi", "there", "guy"]

    def test_attribute_function(self) -> None:
        assert self.x("#p-b2::attr(id)") == ["p-b2"]
        assert self.x(".cool-footer::attr(class)") == ["cool-footer"]
        assert self.x(".cool-footer ::attr(id)") == ["foobar-div", "foobar-span"]
        assert self.x('map[name="dummymap"] ::attr(shape)') == ["circle", "default"]

    def test_nested_selector(self) -> None:
        assert self.sel.css("p").css("b::text").extract() == ["hi", "guy"]
        assert self.sel.css("div").css("area:last-child").extract() == [
            '<area shape="default" id="area-nohref">'
        ]

    def test_pseudoclass_has(self) -> None:
        assert self.x("p:has(b)::text") == ["lorem ipsum text"]


class TestCSSSelectorBytes(TestCSSSelector):
    sel = Selector(body=bytes(HTMLBODY, encoding="utf-8"))


# The attribute name as stored in the tree differs by document type: the HTML
# parser lower-cases it, while the XML parser keeps it as written.
HTML_UPPERCASE_DOC = '<html><body><img SRC="u"/></body></html>'
XML_UPPERCASE_DOC = '<?xml version="1.0"?><root><img SRC="u"/></root>'
XML_LOWERCASE_DOC = '<?xml version="1.0"?><root><img src="u"/></root>'
XML_MIXEDCASE_DOC = '<?xml version="1.0"?><root><img sRc="u"/></root>'
XML_UPPERCASE_ELEMENT = '<img SRC="u"/>'
XML_LOWERCASE_ELEMENT = '<img src="u"/>'
XML_MIXEDCASE_ELEMENT = '<img sRc="u"/>'


@pytest.mark.parametrize(
    ("translator", "query", "xpath"),
    [
        # Lower-case queries keep their exact, pre-existing translation.
        (HTMLTranslator, "img[src]", "descendant-or-self::img[@src]"),
        (HTMLTranslator, "img::attr(src)", "descendant-or-self::img/@src"),
        (GenericTranslator, "img[src]", "descendant-or-self::img[@src]"),
        (GenericTranslator, "img::attr(src)", "descendant-or-self::img/@src"),
        # HTML: both writings lower-case the attribute name, so they translate
        # to the same XPath as their lower-case equivalents.
        (HTMLTranslator, "img[SRC]", "descendant-or-self::img[@src]"),
        (HTMLTranslator, "img::attr(SRC)", "descendant-or-self::img/@src"),
        (HTMLTranslator, "img[sRc]", "descendant-or-self::img[@src]"),
        (HTMLTranslator, "img::attr(sRc)", "descendant-or-self::img/@src"),
        # XML: both writings keep the attribute name as written.
        (GenericTranslator, "img[SRC]", "descendant-or-self::img[@SRC]"),
        (GenericTranslator, "img::attr(SRC)", "descendant-or-self::img/@SRC"),
        (GenericTranslator, "img[sRc]", "descendant-or-self::img[@sRc]"),
        (GenericTranslator, "img::attr(sRc)", "descendant-or-self::img/@sRc"),
    ],
)
def test_attribute_name_case_translation(
    translator: type[GenericTranslator | HTMLTranslator], query: str, xpath: str
) -> None:
    assert translator().css_to_xpath(query) == xpath


@pytest.mark.parametrize(
    ("query", "matched"),
    [
        # HTML matches attribute names case-insensitively: every writing and
        # every query case hits the same lower-cased attribute.
        ("img[SRC]", True),
        ("img::attr(SRC)", True),
        ("img[src]", True),
        ("img::attr(src)", True),
        ("img[sRc]", True),
        ("img::attr(sRc)", True),
    ],
)
def test_html_attribute_name_case_matching(query: str, matched: bool) -> None:
    sel = Selector(text=HTML_UPPERCASE_DOC, type="html")
    assert bool(sel.css(query)) is matched
    # The two writings of a query with the same case agree on the hit, and the
    # pseudo-element returns the attribute value when it matches.
    if "::attr(" in query:
        assert sel.css(query).getall() == (["u"] if matched else [])
    else:
        assert sel.css(query).getall() == (['<img src="u">'] if matched else [])


@pytest.mark.parametrize(
    ("document", "element", "query", "matched"),
    [
        # XML matches attribute names case-sensitively, keeping the name as
        # written: [attr] and ::attr(attr) agree for every query case.
        (XML_UPPERCASE_DOC, XML_UPPERCASE_ELEMENT, "img[SRC]", True),
        (XML_UPPERCASE_DOC, XML_UPPERCASE_ELEMENT, "img::attr(SRC)", True),
        (XML_UPPERCASE_DOC, XML_UPPERCASE_ELEMENT, "img[src]", False),
        (XML_UPPERCASE_DOC, XML_UPPERCASE_ELEMENT, "img::attr(src)", False),
        (XML_LOWERCASE_DOC, XML_LOWERCASE_ELEMENT, "img[SRC]", False),
        (XML_LOWERCASE_DOC, XML_LOWERCASE_ELEMENT, "img::attr(SRC)", False),
        (XML_LOWERCASE_DOC, XML_LOWERCASE_ELEMENT, "img[src]", True),
        (XML_LOWERCASE_DOC, XML_LOWERCASE_ELEMENT, "img::attr(src)", True),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img[sRc]", True),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img::attr(sRc)", True),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img[SRC]", False),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img::attr(SRC)", False),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img[src]", False),
        (XML_MIXEDCASE_DOC, XML_MIXEDCASE_ELEMENT, "img::attr(src)", False),
    ],
)
def test_xml_attribute_name_case_matching(
    document: str, element: str, query: str, matched: bool
) -> None:
    sel = Selector(text=document, type="xml")
    assert bool(sel.css(query)) is matched
    if "::attr(" in query:
        assert sel.css(query).getall() == (["u"] if matched else [])
    else:
        assert sel.css(query).getall() == ([element] if matched else [])


def test_html_attribute_writings_agree_across_cases() -> None:
    """For a fixed attribute, [attr] and ::attr(attr) never contradict each
    other, regardless of the case used in the query."""
    sel = Selector(text=HTML_UPPERCASE_DOC, type="html")
    for case in ("SRC", "src", "sRc", "Src"):
        attrib_selector = sel.css(f"img[{case}]")
        attr_pseudo = sel.css(f"img::attr({case})")
        assert bool(attrib_selector) is bool(attr_pseudo)
        assert attr_pseudo.getall() == ["u"]


def test_xml_attribute_writings_agree_across_cases() -> None:
    """Same contract as the HTML side, but with case-sensitive matching."""
    for document, stored in (
        (XML_UPPERCASE_DOC, "SRC"),
        (XML_LOWERCASE_DOC, "src"),
        (XML_MIXEDCASE_DOC, "sRc"),
    ):
        sel = Selector(text=document, type="xml")
        for case in ("SRC", "src", "sRc", "Src"):
            attrib_selector = sel.css(f"img[{case}]")
            attr_pseudo = sel.css(f"img::attr({case})")
            assert bool(attrib_selector) is bool(attr_pseudo)
            expected = ["u"] if case == stored else []
            assert attr_pseudo.getall() == expected
