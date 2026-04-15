import asyncio
import logging
import time
from types import SimpleNamespace

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchParser, BatchProcessingResult
from raganything.config import RAGAnythingConfig
from raganything.processor import ProcessorMixin
from raganything.raganything import RAGAnything


class _DummyParser:
    def __init__(self, name: str):
        self.name = name

    def check_installation(self):
        return True


def test_update_config_resyncs_parser(monkeypatch):
    def fake_get_parser(name):
        return _DummyParser(name)

    monkeypatch.setattr("raganything.raganything.get_parser", fake_get_parser)

    rag = RAGAnything(config=RAGAnythingConfig(parser="mineru"))
    rag._parser_installation_checked = True

    rag.update_config(parser="docling")

    assert rag.doc_parser.name == "docling"
    assert rag._parser_installation_checked is False


def test_batch_process_async_forwards_kwargs(monkeypatch):
    parser = BatchParser(parser_type="mineru", skip_installation_check=True)
    captured = {}

    def fake_process_batch(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(parser, "process_batch", fake_process_batch)

    result = asyncio.run(
        parser.process_batch_async(
            file_paths=["doc.pdf"],
            output_dir="out",
            parse_method="auto",
            recursive=False,
            dry_run=True,
            custom_flag=True,
        )
    )

    assert result == "ok"
    assert captured["custom_flag"] is True
    assert captured["dry_run"] is True


def test_batch_process_timeout_is_tracked(monkeypatch, tmp_path):
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    parser = BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        timeout_per_file=0.1,
        show_progress=False,
    )

    def slow_process_single_file(*args, **kwargs):
        time.sleep(0.3)
        return True, str(file_path), None

    monkeypatch.setattr(parser, "process_single_file", slow_process_single_file)

    result = parser.process_batch(
        file_paths=[str(file_path)],
        output_dir=str(tmp_path / "out"),
        recursive=False,
    )

    assert result.failed_files == [str(file_path)]
    assert "Timed out after" in result.errors[str(file_path)]


class _DummyBatchRunner(BatchMixin):
    def __init__(self):
        self.config = RAGAnythingConfig()
        self.logger = logging.getLogger("test-batch-runner")
        self.parsed = []
        self.processed = []

    async def _ensure_lightrag_initialized(self):
        return None

    async def parse_document(self, file_path: str, **kwargs):
        self.parsed.append((file_path, kwargs))
        return ([{"type": "text", "text": f"parsed:{file_path}"}], f"doc-{file_path}")

    async def process_document_complete(self, file_path: str, **kwargs):
        self.processed.append((file_path, kwargs))


def test_process_documents_with_rag_batch_reuses_parse_results(monkeypatch, tmp_path):
    runner = _DummyBatchRunner()
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = asyncio.run(
        runner.process_documents_with_rag_batch(
            file_paths=[str(file_path)],
            output_dir=str(tmp_path / "out"),
            recursive=False,
            show_progress=False,
        )
    )

    assert runner.parsed == [
        (
            str(file_path),
            {
                "output_dir": str(tmp_path / "out"),
                "parse_method": runner.config.parse_method,
                "display_stats": False,
            },
        )
    ]
    assert len(runner.processed) == 1
    processed_kwargs = runner.processed[0][1]
    assert processed_kwargs["parsed_content_list"][0]["text"] == f"parsed:{file_path}"
    assert processed_kwargs["parsed_doc_id"] == f"doc-{file_path}"
    assert result["successful_rag_files"] == 1


class _DummyProcessorAPI(ProcessorMixin):
    def __init__(self):
        self.config = RAGAnythingConfig()
        self.logger = logging.getLogger("test-processor-api")
        self.lightrag = None

    async def _ensure_lightrag_initialized(self):
        return {"success": False, "error": "init failed"}


def test_process_document_complete_lightrag_api_handles_init_failure():
    processor = _DummyProcessorAPI()

    result = asyncio.run(
        processor.process_document_complete_lightrag_api("sample.pdf")
    )

    assert result is False
