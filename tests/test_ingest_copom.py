from pathlib import Path
import pytest
from langchain_core.documents import Document

from rag.ingest_copom import (
    build_copom_chunks,
    clean_text,
    extract_meeting_number,
)


class TestCleanText:
    def test_collapses_horizontal_whitespace_while_preserving_newlines(self) -> None:
        raw = "Taxa   Selic\t\tfixada   em 14,25%   ao ano.\nSegundo\t  parágrafo."
        cleaned = clean_text(raw)
        assert cleaned == "Taxa Selic fixada em 14,25% ao ano.\nSegundo parágrafo."

    def test_strips_bacen_page_number_footer(self) -> None:
        raw = (
            "Texto antes da quebra de página.\n"
            "  14  \n"
            "bcb.gov.br\n"
            "Texto após a quebra de página."
        )
        cleaned = clean_text(raw)
        assert "bcb.gov.br" not in cleaned
        assert "\n14\n" not in cleaned
        assert (
            cleaned
            == "Texto antes da quebra de página.\n\nTexto após a quebra de página."
        )

    def test_strips_copom_running_header(self) -> None:
        raw = (
            "280ª Reunião\n"
            "COPOM\n"
            "10 e 11 de dezembro de 2025\n"
            "B) Riscos para o cenário de inflação"
        )
        cleaned = clean_text(raw)
        assert "280ª Reunião" not in cleaned
        assert "COPOM" not in cleaned
        assert cleaned == "B) Riscos para o cenário de inflação"

    def test_collapses_excessive_blank_lines_and_trims_edges(
        self, sample_raw_copom_text: str
    ) -> None:
        cleaned = clean_text(sample_raw_copom_text)
        assert "\n\n\n" not in cleaned
        assert not cleaned.startswith((" ", "\n"))
        assert not cleaned.endswith((" ", "\n"))
        assert "bcb.gov.br" not in cleaned
        assert "282ª Reunião" not in cleaned
        assert (
            "1. O ambiente externo permanece incerto em função da política monetária nos EUA."
            in cleaned
        )
        assert (
            "2. No cenário doméstico, os indicadores de atividade apresentam moderação."
            in cleaned
        )


class TestChunkMetadataExtraction:
    def test_extract_meeting_number_from_filename(self) -> None:
        assert extract_meeting_number(Path("data/Copom282.pdf")) == "282"
        assert extract_meeting_number(Path("Copom276_ata.pdf")) == "276"

    def test_extract_meeting_number_raises_when_digits_missing(self) -> None:
        with pytest.raises(ValueError, match="No meeting number"):
            extract_meeting_number(Path("data/Copom_latest.pdf"))

    def test_build_copom_chunks_populates_meeting_start_index_and_deterministic_ids(
        self,
    ) -> None:
        paragraph = (
            "O Comitê de Política Monetária decidiu manter a taxa básica de juros "
            "em linha com a convergência da inflação para a meta no horizonte relevante. "
        )
        long_text = "281ª Reunião\nCOPOM\nJaneiro 2026\n" + (paragraph * 15)
        doc = Document(
            page_content=long_text,
            metadata={"source": "/workspace/data/Copom281.pdf", "page": 0},
        )

        chunks, ids = build_copom_chunks(
            docs=[doc],
            meeting="281",
            chunk_size=400,
            chunk_overlap=80,
        )

        assert len(chunks) > 1
        assert len(chunks) == len(ids)

        start_indices = [c.metadata["start_index"] for c in chunks]
        assert start_indices[0] == 0
        assert start_indices == sorted(start_indices)
        assert len(set(start_indices)) == len(start_indices)

        for chunk, chunk_id in zip(chunks, ids):
            assert chunk.metadata["meeting"] == "281"
            assert isinstance(chunk.metadata["start_index"], int)
            assert (
                chunk_id
                == f"file_Copom281.pdf_start_idx_{chunk.metadata['start_index']}"
            )
            assert "281ª Reunião" not in chunk.page_content
