from unittest.mock import MagicMock
import pytest


def _install_offline_stubs() -> None:
    """Prevent heavy model downloads or DB connections when importing rag modules."""
    import langchain.chat_models as chat_models
    import langchain_chroma as chroma_mod
    import langchain_huggingface as hf_mod

    dummy_llm = MagicMock(name="DummyChatModel")
    dummy_llm.with_structured_output.return_value = MagicMock(name="DummyStructuredLLM")
    dummy_llm.bind_tools.return_value = MagicMock(name="DummyToolLLM")

    chat_models.init_chat_model = MagicMock(return_value=dummy_llm)
    hf_mod.HuggingFaceEmbeddings = MagicMock(
        return_value=MagicMock(name="DummyEmbeddings")
    )
    chroma_mod.Chroma = MagicMock(return_value=MagicMock(name="DummyChroma"))


_install_offline_stubs()


@pytest.fixture
def sample_raw_copom_text() -> str:
    return (
        "282ª Reunião\n"
        "COPOM\n"
        "Notas da 282ª Reunião do Comitê de Política Monetária\n"
        "A) Atualização da conjuntura econômica   e   cenário básico\n\n\n\n"
        "1.   O ambiente externo permanece   incerto em função da política monetária nos EUA.\n"
        "   3   \n"
        "bcb.gov.br\n"
        "282ª Reunião\n"
        "COPOM\n"
        "17 e 18 de março de 2026\n"
        "2.   No cenário doméstico, os indicadores de atividade apresentam moderação.\n"
    )
