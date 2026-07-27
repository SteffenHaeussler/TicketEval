import httpx
import pytest

from ticketflow import config
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.models import ActionType, Classification, Ticket, TicketCategory

pytestmark = pytest.mark.ollama


BILLING_TICKET = Ticket(
    id="t-billing",
    customer_email="jo@example.com",
    subject="Refund for double charge",
    body=(
        "I was charged twice for my subscription this month. "
        "Please refund one of the charges."
    ),
)

TECH_TICKET = Ticket(
    id="t-tech",
    customer_email="sam@example.com",
    subject="App crashes on login",
    body=(
        "Every time I try to log in the app crashes immediately. "
        "I'm on the latest version."
    ),
)

TICKETS = [BILLING_TICKET, TECH_TICKET]
TICKET_IDS = ["billing", "tech"]


@pytest.fixture(scope="module")
def ollama_ready():
    try:
        response = httpx.get(f"{config.OLLAMA_ENDPOINT}/api/version", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(
            f"Ollama not reachable at {config.OLLAMA_ENDPOINT} ({exc}); "
            f"start it with `ollama serve` and `ollama pull {config.OLLAMA_MODEL}`."
        )


@pytest.fixture
async def agent(ollama_ready):
    async with OllamaAgent() as instance:
        yield instance


@pytest.mark.parametrize("ticket", TICKETS, ids=TICKET_IDS)
async def test_classify_returns_valid_classification(agent, ticket):
    result = await agent.classify(ticket)
    assert isinstance(result, Classification)
    assert result.category in set(TicketCategory)
    assert 0.0 <= result.confidence <= 1.0
    assert result.model == "primary"


@pytest.mark.parametrize("ticket", TICKETS, ids=TICKET_IDS)
async def test_draft_reply_returns_valid_draft(agent, ticket):
    classification = await agent.classify(ticket)
    draft = await agent.draft_reply(ticket, classification)
    assert draft.reply_text.strip()
    assert 0.0 <= draft.confidence <= 1.0
    assert draft.model == "primary"
    if draft.action.type == ActionType.REFUND:
        assert draft.action.refund_amount is not None
        assert draft.action.refund_amount > 0
