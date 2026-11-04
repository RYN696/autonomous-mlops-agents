"""Bot Discord : les messages de l'interface arrivent aussi sur Discord, avec des boutons.

Facultatif : sans DISCORD_BOT_TOKEN, rien ne change (tout reste sur la page web).

- #mlops-alertes (DISCORD_ALERTS_CHANNEL_ID) : incidents, données corrompues, budget dépassé ;
- #mlops-agents  (DISCORD_AGENTS_CHANNEL_ID) : réentraînements, promotions, questions.
  Un seul salon renseigné : tout y va.

Un parcours (request_id ou incident_id) = un seul message Discord, complété ligne par
ligne (cause → action → résultat) au lieu d'un message par étape.

Boutons :
- liens          : Rapport (parcours dans l'interface), Voir les données, Tableau de bord (Grafana) ;
- interactifs    : Valider / Refuser, Revenir à vN (avec confirmation), Ignorer.
  Ils appellent HumanActions, comme la page web, et sont retirés une fois utilisés.

Le bot tourne dans son propre fil (boucle asyncio) dans le processus de l'orchestrateur.
Il passe par la passerelle Discord (connexion sortante) : pas besoin d'URL publique.
Les clics sont reconnus par leur custom_id ("mlops:<verbe>:<arguments>"), ce qui marche
encore après un redémarrage (le bouton répond alors que la question a expiré).
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

from agents.orchestrator.human import HumanActionError, HumanActions
from agents.orchestrator.notifier import Message
from common import config

PREFIX = "mlops"
ALERT_LEVELS = ("alert",)                 # vont dans le salon des alertes
MENTION_LEVELS = ("alert", "question")    # précédés de DISCORD_ALERT_MENTION
INTERACTIVE = ("approve", "refuse", "rollback", "ignore")
COLORS = {"info": 0x3B82F6, "success": 0x22C55E, "warning": 0xF59E0B, "alert": 0xEF4444, "question": 0xA855F7}
MAX_DESCRIPTION = 3900                    # limite Discord : 4096 caractères


@dataclass(frozen=True)
class ButtonSpec:
    label: str
    style: str                  # success, danger, secondary, link
    custom_id: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class Reply:
    """Réponse à un clic, visible seulement par la personne qui a cliqué."""
    text: str
    key: str | None = None
    consumed: tuple[str, ...] = ()       # boutons à retirer du message du parcours
    confirm: ButtonSpec | None = None    # second bouton à presser (geste risqué)


def verb(action: str) -> str:
    return action.split(":", 1)[0]


def button_specs(actions: Iterable[str], key: str | None, ui_url: str = config.UI_PUBLIC_URL,
                 grafana_url: str = config.GRAFANA_PUBLIC_URL) -> list[ButtonSpec]:
    k = key or "-"
    specs: list[ButtonSpec] = []
    for action in actions:
        name, *args = action.split(":")
        if name == "approve" and key:
            specs.append(ButtonSpec("Valider", "success", f"{PREFIX}:approve:{key}"))
        elif name == "refuse" and key:
            specs.append(ButtonSpec("Refuser", "danger", f"{PREFIX}:refuse:{key}"))
        elif name == "rollback" and len(args) == 2:
            target, current = args
            specs.append(ButtonSpec(f"Revenir à v{target}", "danger", f"{PREFIX}:rollback:{target}:{current}:{k}"))
        elif name == "ignore":
            specs.append(ButtonSpec("Ignorer", "secondary", f"{PREFIX}:ignore:{k}"))
        elif name == "report" and key:
            specs.append(ButtonSpec("Rapport", "link", url=f"{ui_url}/parcours/{key}"))
        elif name == "data" and key:
            specs.append(ButtonSpec("Voir les données", "link", url=f"{ui_url}/parcours/{key}"))
        elif name == "costs":
            specs.append(ButtonSpec("Rapport de coûts", "link", url=f"{ui_url}/couts"))
        elif name == "dashboard":
            specs.append(ButtonSpec("Tableau de bord", "link", url=grafana_url))
    return specs[:5]  # une rangée de boutons


def dispatch(human: HumanActions, custom_id: str, who: str) -> Reply | None:
    """Exécute le geste d'un bouton. None : bouton qui ne vient pas de ce bot."""
    parts = custom_id.split(":")
    if len(parts) < 3 or parts[0] != PREFIX:
        return None
    name, args = parts[1], parts[2:]

    def key(value: str) -> str | None:
        return None if value == "-" else value

    if name in ("approve", "refuse") and len(args) == 1:
        request_id = args[0]
        if not human.answer(request_id, name == "approve", who, source="discord"):
            return Reply("Cette question a déjà une réponse ou a expiré (orchestrateur redémarré).",
                         request_id, ("approve", "refuse"))
        text = ("Validé : l'orchestrateur lance le réentraînement." if name == "approve"
                else "Refusé : rien n'est lancé, la production ne change pas.")
        return Reply(text, request_id, ("approve", "refuse"))
    if name == "rollback" and len(args) == 3:
        target, current, k = args
        return Reply(f"Remettre v{target} en production à la place de v{current} ?", key(k),
                     confirm=ButtonSpec(f"Confirmer : revenir à v{target}", "danger",
                                        f"{PREFIX}:rollback-ok:{target}:{current}:{k}"))
    if name == "rollback-ok" and len(args) == 3:
        target, current, k = args
        try:
            return Reply(human.rollback(target, current, key(k), who, source="discord"), key(k), ("rollback",))
        except HumanActionError as exc:
            return Reply(f"Retour impossible : {exc}", key(k), ("rollback",) if exc.stale else ())
    if name == "ignore" and len(args) == 1:
        return Reply(human.ignore(key(args[0]), who, source="discord"), key(args[0]), ("ignore",))
    return None


def render(key: str | None, lines: list[str], level: str) -> dict[str, Any]:
    """Contenu de l'embed d'un parcours : les lignes les plus anciennes sautent si c'est trop long."""
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        size += len(line) + 1
        if size > MAX_DESCRIPTION:
            kept.append("…")
            break
        kept.append(line)
    return {"title": f"Parcours {key}" if key else "Pipeline taxi-duration",
            "description": "\n".join(reversed(kept)), "color": COLORS.get(level, COLORS["info"]),
            "footer": f"{config.MODEL_NAME} · agents MLOps"}


@dataclass
class Post:
    """Message Discord d'un parcours, complété à chaque étape."""
    key: str | None
    channel_id: int
    lines: list[str]
    level: str
    actions: tuple[str, ...]
    updated: float
    message: Any = None


class DiscordRelay:
    def __init__(self, human: HumanActions, token: str = config.DISCORD_BOT_TOKEN,
                 alerts_channel: str = config.DISCORD_ALERTS_CHANNEL_ID,
                 agents_channel: str = config.DISCORD_AGENTS_CHANNEL_ID,
                 mention: str = config.DISCORD_ALERT_MENTION, ui_url: str = config.UI_PUBLIC_URL,
                 grafana_url: str = config.GRAFANA_PUBLIC_URL):
        import discord  # seulement si Discord est configuré

        self.discord = discord
        self.human = human
        self.token = token
        self.alerts_channel = int(alerts_channel or agents_channel)
        self.agents_channel = int(agents_channel or alerts_channel)
        self.mention = mention.strip()
        self.ui_url = ui_url.rstrip("/")
        self.grafana_url = grafana_url
        self.links_as_text = False   # Discord refuse parfois les boutons-liens vers localhost
        self.posts: dict[tuple[str, int], Post] = {}
        self.status = "connexion…"
        self.failed = False
        self.loop = asyncio.new_event_loop()
        self._lock: asyncio.Lock | None = None
        self.client = discord.Client(intents=discord.Intents.default())
        self.client.event(self.on_ready)
        self.client.event(self.on_interaction)

    # ------------------------------------------------ démarrage
    def start(self) -> None:
        threading.Thread(target=self._run, name="discord", daemon=True).start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.client.start(self.token))
        except self.discord.LoginFailure:
            self.status = "jeton refusé par Discord (DISCORD_BOT_TOKEN)"
        except Exception as exc:
            self.status = f"arrêté : {type(exc).__name__}: {exc}"
        self.failed = True
        print(f"Discord : {self.status}")

    async def on_ready(self) -> None:
        missing = []
        for channel_id in {self.alerts_channel, self.agents_channel}:
            try:
                await self.channel(channel_id)
            except Exception:
                missing.append(str(channel_id))
        self.status = (f"salon introuvable ou interdit au bot : {', '.join(missing)}" if missing
                       else f"connecté : {self.client.user}")
        print(f"Discord : {self.status}")

    def state(self) -> dict[str, Any]:
        """Pastille de la vue d'ensemble."""
        ok = self.status.startswith("connecté")
        return {"name": "Discord", "ok": ok or not self.failed, "warn": not ok, "detail": self.status}

    async def channel(self, channel_id: int):
        return self.client.get_channel(channel_id) or await self.client.fetch_channel(channel_id)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------ messages (appelé depuis les autres fils)
    def post(self, message: Message) -> None:
        if self.failed:
            return
        future = asyncio.run_coroutine_threadsafe(self._post(message), self.loop)
        future.add_done_callback(self._report)

    @staticmethod
    def _report(future) -> None:
        if not future.cancelled() and future.exception():
            exc = future.exception()
            print(f"Discord : message non envoyé ({type(exc).__name__}: {exc})")

    async def _post(self, message: Message) -> None:
        await self.client.wait_until_ready()
        channel_id = self.alerts_channel if message.level in ALERT_LEVELS else self.agents_channel
        now = time.time()
        line = f"<t:{int(now)}:T> {message.text}"
        async with self.lock:
            post = self.posts.get((message.key, channel_id)) if message.key else None
            if post and now - post.updated < config.DISCORD_THREAD_HOURS * 3600:
                post.lines.append(line)
                post.level = message.level
                # sans nouveaux boutons, on garde les liens du message (pas un ancien Valider/Refuser)
                post.actions = message.actions or tuple(a for a in post.actions if verb(a) not in INTERACTIVE)
                post.updated = now
                await self._edit(post)
                return
            post = Post(message.key, channel_id, [line], message.level, message.actions, now)
            content = self.mention if self.mention and message.level in MENTION_LEVELS else None
            post.message = await self._send(await self.channel(channel_id), post, content)
            if message.key:
                self.posts[(message.key, channel_id)] = post
                self._forget_old(now)

    def _forget_old(self, now: float) -> None:
        limit = now - config.DISCORD_THREAD_HOURS * 3600
        for k in [k for k, p in self.posts.items() if p.updated < limit]:
            del self.posts[k]

    # ------------------------------------------------ rendu
    def _view(self, specs: list[ButtonSpec]):
        if not specs:
            return None
        d = self.discord
        view = d.ui.View(timeout=None)
        for s in specs:
            if s.style == "link":
                view.add_item(d.ui.Button(label=s.label, style=d.ButtonStyle.link, url=s.url))
            else:
                view.add_item(d.ui.Button(label=s.label, style=getattr(d.ButtonStyle, s.style), custom_id=s.custom_id))
        return view

    def _render(self, post: Post):
        specs = button_specs(post.actions, post.key, self.ui_url, self.grafana_url)
        e = render(post.key, post.lines, post.level)
        if self.links_as_text:
            links = [f"🔗 [{s.label}]({s.url})" for s in specs if s.style == "link"]
            e["description"] += ("\n" + " · ".join(links)) if links else ""
            specs = [s for s in specs if s.style != "link"]
        embed = self.discord.Embed(title=e["title"], description=e["description"], color=e["color"])
        embed.set_footer(text=e["footer"])
        return embed, self._view(specs)

    def _has_links(self, post: Post) -> bool:
        return any(s.style == "link" for s in button_specs(post.actions, post.key, self.ui_url, self.grafana_url))

    async def _send(self, channel, post: Post, content: str | None):
        for _ in range(2):
            embed, view = self._render(post)
            kwargs = {"content": content, "embed": embed}
            if view is not None:
                kwargs["view"] = view
            try:
                return await channel.send(**kwargs)
            except self.discord.HTTPException as exc:
                if not self._links_rejected(exc, post):
                    raise

    async def _edit(self, post: Post) -> None:
        if post.message is None:
            return
        for _ in range(2):
            embed, view = self._render(post)
            try:
                await post.message.edit(embed=embed, view=view)
                return
            except self.discord.HTTPException as exc:
                if not self._links_rejected(exc, post):
                    raise

    def _links_rejected(self, exc, post: Post) -> bool:
        """Erreur 400 avec des boutons-liens : on réessaie avec les liens écrits dans le texte."""
        if exc.status == 400 and not self.links_as_text and self._has_links(post):
            print(f"Discord : boutons-liens refusés ({exc.text}) ; liens écrits dans le message")
            self.links_as_text = True
            return True
        return False

    # ------------------------------------------------ clics
    async def on_interaction(self, interaction) -> None:
        custom_id = (interaction.data or {}).get("custom_id", "")
        if not custom_id.startswith(f"{PREFIX}:"):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        who = f"{interaction.user.display_name} (Discord)"
        try:
            reply = await asyncio.to_thread(dispatch, self.human, custom_id, who)
        except Exception as exc:
            await interaction.followup.send(f"Erreur : {type(exc).__name__}: {exc}", ephemeral=True)
            return
        if reply is None:
            await interaction.followup.send("Bouton inconnu.", ephemeral=True)
            return
        kwargs = {"ephemeral": True}
        view = self._view([reply.confirm]) if reply.confirm else None
        if view is not None:
            kwargs["view"] = view
        await interaction.followup.send(reply.text, **kwargs)
        if reply.consumed:
            await self._consume(reply, interaction.message)

    async def _consume(self, reply: Reply, clicked) -> None:
        """Retire les boutons utilisés du message du parcours."""
        async with self.lock:
            posts = [p for p in self.posts.values() if reply.key and p.key == reply.key]
            for p in posts:
                kept = tuple(a for a in p.actions if verb(a) not in reply.consumed)
                if kept != p.actions:
                    p.actions = kept
                    await self._edit(p)
        if not posts and clicked is not None and not clicked.flags.ephemeral:
            try:  # message d'avant un redémarrage de l'orchestrateur : on retire ses boutons
                await clicked.edit(view=None)
            except Exception:
                pass


def from_config(human: HumanActions) -> DiscordRelay | None:
    """Le relais si Discord est configuré, sinon None (avec la raison dans les journaux)."""
    if not config.DISCORD_BOT_TOKEN:
        return None
    if not (config.DISCORD_ALERTS_CHANNEL_ID or config.DISCORD_AGENTS_CHANNEL_ID):
        print("Discord : DISCORD_BOT_TOKEN sans salon (DISCORD_ALERTS_CHANNEL_ID / DISCORD_AGENTS_CHANNEL_ID)")
        return None
    try:
        return DiscordRelay(human)
    except ValueError:
        print("Discord : les identifiants de salon doivent être des nombres (clic droit > Copier l'identifiant)")
        return None
