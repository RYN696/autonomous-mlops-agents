import threading

from agents.quality.watch import watch
from common.events import Event, EventType


class PromotingBus:
    """Envoie un model_promoted dès que l'agent écoute."""

    def __init__(self):
        self.listening = threading.Event()

    def listen(self, handler, types=None):
        assert types == [EventType.MODEL_PROMOTED]
        self.listening.set()
        handler(Event(type=EventType.MODEL_PROMOTED, source="agent_retrain",
                      payload={"challenger_version": "7"}))


def test_controle_au_demarrage_puis_a_chaque_promotion():
    stop, calls, bus = threading.Event(), [], PromotingBus()

    def check():
        calls.append(1)
        if len(calls) == 1:
            bus.listening.wait(2)  # la promotion arrive pendant le 1er contrôle
        else:
            stop.set()

    watch(check, bus, interval_seconds=60, stop=stop)  # sans la promotion : 60 s d'attente
    assert len(calls) == 2


def test_un_controle_en_erreur_n_arrete_pas_le_service():
    stop, calls = threading.Event(), []

    class SilentBus:
        def listen(self, handler, types=None):
            pass

    def check():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("MLflow injoignable")
        stop.set()

    watch(check, SilentBus(), interval_seconds=0.01, stop=stop)
    assert len(calls) == 2
