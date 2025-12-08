import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from scidata.components.search import HybridSearchApp, BaseDocument


class TestBaseDocument:
    """Test BaseDocument dataclass"""

    def test_base_document_with_valid_id(self):
        """Test creating BaseDocument with valid id"""
        doc = BaseDocument(id="123", embedding=None)
        assert doc.id == "123"

    def test_base_document_with_none_id_raises_error(self):
        """Test that BaseDocument raises error if id is None"""
        with pytest.raises((ValueError, TypeError)):
            BaseDocument(id=None)


class TestHybridSearchApp:
    """Test HybridSearchApp class"""

    @patch('scidata.components.search.AsyncOpenSearch')
    @patch('scidata.components.search.AsyncOpenAI')
    def test_init(self, mock_openai, mock_opensearch):
        """Test HybridSearchApp initialization"""
        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )
        
        assert app.db_index == "test-index"
        assert app.field == "content"
        assert app.model == "text-embedding-3-small"
        assert app.opensearch_client is None
        assert app.openai_client is None

    @pytest.mark.asyncio
    @patch('scidata.components.search.AsyncOpenSearch')
    @patch('scidata.components.search.AsyncOpenAI')
    async def test_context_manager_enter(self, mock_openai, mock_opensearch):
        """Test context manager __enter__ method"""
        mock_os_client = AsyncMock()
        mock_os_client.indices.exists = AsyncMock(return_value=False)
        mock_os_client.close = AsyncMock()
        mock_opensearch.return_value = mock_os_client
        
        mock_openai_client = AsyncMock()
        mock_openai_client.close = AsyncMock()
        mock_openai.return_value = mock_openai_client

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        async with app as context_app:
            assert context_app.opensearch_client is not None
            assert context_app.openai_client is not None

    @pytest.mark.asyncio
    @patch('scidata.components.search.AsyncOpenSearch')
    @patch('scidata.components.search.AsyncOpenAI')
    async def test_normalize_embedding(self, mock_openai, mock_opensearch):
        """Test embedding normalization"""
        mock_os_client = AsyncMock()
        mock_os_client.indices.exists = AsyncMock(return_value=False)
        mock_os_client.close = AsyncMock()
        mock_opensearch.return_value = mock_os_client
        
        mock_openai_client = AsyncMock()
        mock_openai_client.close = AsyncMock()
        mock_openai.return_value = mock_openai_client

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        async with app:
            # Test normalization
            embedding = [3.0, 4.0]
            normalized = app.normalize(embedding)
            
            # Min-max normalization : (x - min) / (max - min)
            assert abs(normalized[0] - 0.0) < 0.001
            assert abs(normalized[1] - 1.0) < 0.001

    @pytest.mark.asyncio
    @patch('scidata.components.search.AsyncOpenSearch')
    @patch('scidata.components.search.AsyncOpenAI')
    async def test_normalize_embedding_zero_vector(self, mock_openai, mock_opensearch):
        """Test normalization of zero vector"""
        mock_os_client = AsyncMock()
        mock_os_client.indices.exists = AsyncMock(return_value=False)
        mock_os_client.close = AsyncMock()
        mock_opensearch.return_value = mock_os_client
        
        mock_openai_client = AsyncMock()
        mock_openai_client.close = AsyncMock()
        mock_openai.return_value = mock_openai_client

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        async with app:
            # Zero vector should return as-is
            zero_embedding = [0.0, 0.0, 0.0]
            normalized = app.normalize(zero_embedding)
            assert normalized == [0.5, 0.5, 0.5] 


class TestHybridSearchIntegration:
    """Integration tests for HybridSearchApp"""

    def test_opensearch_index_body_structure(self):
        """Test that index body has correct structure"""
        index_body = HybridSearchApp.OPENSEARCH_INDEX_BODY
        
        assert "settings" in index_body
        assert "mappings" in index_body
        assert "index" in index_body["settings"]
        assert "properties" in index_body["mappings"]
        assert "embedding" in index_body["mappings"]["properties"]
        assert "content" in index_body["mappings"]["properties"]

    def test_knn_settings_enabled(self):
        """Test that k-NN is enabled in index settings"""
        index_body = HybridSearchApp.OPENSEARCH_INDEX_BODY
        
        assert index_body["settings"]["index"]["knn"] is True
        assert index_body["settings"]["index"]["knn.algo_param.ef_search"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
