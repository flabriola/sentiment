from unittest.mock import patch
from project import fetch_articles, spacer, get_sentiment

@patch('os.get_terminal_size')
def test_spacer(mock):
    mock.return_value.columns = 80
    test_string = "Test"
    result = spacer(test_string)
    
    assert len(result) == len(test_string) or len(result) == 80
    assert all(char == "-" for char in result)
    
@patch('os.get_terminal_size')
def test_spacer_type(mock):
    mock.return_value.columns = 80
    result = spacer("Test")
    assert isinstance(result, str)

@patch('project.requests.get')
def test_fetch_articles(mock):
    mock_response = {
        'status': 'ok',
        'articles': [
            {'title': 'Article 1', 'description': 'Description 1'},
            {'title': 'Article 2', 'description': 'Description 2'}
        ]
    }
    mock.return_value.json.return_value = mock_response

    q = 'test'
    articles = fetch_articles(q)

    assert len(articles) == 2
    assert articles[0]['title'] == 'Article 1'
    assert articles[1]['description'] == 'Description 2'

@patch('project.model.predict')
@patch('project.tokenizer.texts_to_sequences')
def test_get_sentiment(mock_texts_to_sequences, mock_predict):
    mock_texts_to_sequences.return_value = [[1, 2, 3, 4, 5]]
    mock_predict.return_value = [[0.1, 0.1, 0.8]]

    article = "Test Article"
    sentiment = get_sentiment(article)

    assert sentiment == 'positive'

@patch('project.requests.get')
def test_fetch_articles_with_error(mock):
    mock_response = {'status': 'error'}
    mock.return_value.json.return_value = mock_response

    q = 'test'
    result = fetch_articles(q)

    assert result == 'error'
