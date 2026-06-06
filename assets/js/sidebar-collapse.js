(function() {
  var attr = "sidebar-collapsed";
  var key = "solanian-sidebar-collapsed";
  var mql = window.matchMedia("(min-width: 850px)");

  function readSaved() {
    try {
      return window.localStorage.getItem(key) === "true";
    } catch (error) {
      return false;
    }
  }

  function writeSaved(value) {
    try {
      window.localStorage.setItem(key, value ? "true" : "false");
    } catch (error) {
      // Storage can be unavailable in private browsing modes.
    }
  }

  function updateButton(active) {
    var button = document.getElementById("sidebar-collapse-toggle");
    var icon = button ? button.querySelector("i") : null;

    if (!button) {
      return;
    }

    button.setAttribute("aria-expanded", active ? "false" : "true");
    button.setAttribute("aria-label", active ? "Expand sidebar" : "Collapse sidebar");
    button.setAttribute("title", active ? "Expand sidebar" : "Collapse sidebar");

    if (icon) {
      icon.className = active ? "fas fa-angle-double-right" : "fas fa-angle-double-left";
    }
  }

  function setCollapsed(value) {
    var active = value && mql.matches;

    if (active) {
      document.body.setAttribute(attr, "");
    } else {
      document.body.removeAttribute(attr);
    }

    updateButton(active);
  }

  function init() {
    var button = document.getElementById("sidebar-collapse-toggle");

    if (!button) {
      return;
    }

    setCollapsed(readSaved());

    button.addEventListener("click", function() {
      var next = !document.body.hasAttribute(attr);
      writeSaved(next);
      setCollapsed(next);
    });

    var onViewportChange = function() {
      setCollapsed(readSaved());
    };

    if (mql.addEventListener) {
      mql.addEventListener("change", onViewportChange);
    } else if (mql.addListener) {
      mql.addListener(onViewportChange);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
