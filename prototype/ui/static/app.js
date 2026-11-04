// Heures locales, durées relatives et rafraîchissement des zones [data-live] sans recharger la page.
(function () {
  const fmtTime = new Intl.DateTimeFormat("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const fmtDay = new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "2-digit" });

  function ago(date) {
    const s = Math.max(0, Math.round((Date.now() - date) / 1000));
    if (s < 60) return "il y a " + s + " s";
    if (s < 3600) return "il y a " + Math.floor(s / 60) + " min";
    if (s < 86400) return "il y a " + Math.floor(s / 3600) + " h";
    return "il y a " + Math.floor(s / 86400) + " j";
  }

  function paintTimes(root) {
    root.querySelectorAll("[data-ts]").forEach((el) => {
      const d = new Date(el.dataset.ts);
      if (isNaN(d)) return;
      const today = new Date().toDateString() === d.toDateString();
      if (el.hasAttribute("data-rel")) el.textContent = ago(d);
      else el.textContent = (today ? "" : fmtDay.format(d) + " ") + fmtTime.format(d);
      el.title = d.toLocaleString("fr-FR");
    });
  }

  // Prénom gardé d'une visite à l'autre pour signer les validations
  function wireWho(root) {
    root.querySelectorAll("form.answer, form[data-sign]").forEach((form) => {
      form.addEventListener("submit", () => {
        const who = document.getElementById("who");
        const field = form.querySelector("input[type=hidden][name=who]");
        if (field && who && who.value) field.value = who.value;
      });
    });
  }
  const who = document.getElementById("who");
  if (who) {
    try { who.value = localStorage.getItem("who") || ""; } catch (e) {}
    who.addEventListener("input", () => { try { localStorage.setItem("who", who.value); } catch (e) {} });
  }

  async function refresh() {
    const zones = document.querySelectorAll("[data-live]");
    if (!zones.length || document.hidden) return;
    // ne pas remplacer une zone pendant qu'on y tape
    const busy = document.activeElement && document.activeElement.closest("[data-live]");
    try {
      const resp = await fetch(location.href, { headers: { "X-Live": "1" } });
      if (!resp.ok) return;
      const doc = new DOMParser().parseFromString(await resp.text(), "text/html");
      zones.forEach((zone) => {
        if (zone === busy) return;
        const fresh = doc.getElementById(zone.id);
        if (!fresh || fresh.innerHTML === zone.innerHTML) return;
        const open = [...zone.querySelectorAll("details[open]")].map((d) => d.id);
        zone.innerHTML = fresh.innerHTML;
        open.forEach((id) => { const d = document.getElementById(id); if (d) d.open = true; });
        paintTimes(zone);
        wireWho(zone);
      });
      const r = document.getElementById("refreshed");
      if (r) r.textContent = "mis à jour " + fmtTime.format(new Date());
    } catch (e) {
      const r = document.getElementById("refreshed");
      if (r) r.textContent = "hors ligne";
    }
  }

  paintTimes(document);
  wireWho(document);
  setInterval(refresh, 5000);
  setInterval(() => document.querySelectorAll("[data-rel]").forEach((el) => paintTimes(el.parentElement)), 15000);
})();
