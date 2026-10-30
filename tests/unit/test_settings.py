from config.settings import EnvironmentSettings, get_limits, get_model_names


def test_limits_file_loads_and_chunk_sizes_are_ordered():
    chunking_limits = get_limits().chunking

    assert chunking_limits.min_tokens < chunking_limits.max_tokens


def test_embedding_batch_respects_the_pinecone_limit():
    assert get_limits().indexing.embedding_batch_size <= 96


def test_model_names_file_loads():
    assert get_model_names().embedding_dimension == 1024


def test_enabled_faults_ignores_spaces_and_empty_entries():
    settings = EnvironmentSettings(
        pinecone_api_key="test-key",
        pinecone_index="test-index",
        faults=" pinecone, ,mcp ",
    )

    assert settings.enabled_faults == {"pinecone", "mcp"}
