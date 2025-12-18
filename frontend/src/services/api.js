class ApiService {
  constructor(backendUrl = 'http://localhost:8000') {
    this.backendUrl = backendUrl;
  }

  async queryGlobal(question, includeCitations = true) {
    try {
      const response = await fetch(`${this.backendUrl}/api/v1/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: question,
          include_citations: includeCitations
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in global query:', error);
      throw error;
    }
  }

  async querySelected(question, selectedText, pageContext = '') {
    try {
      const response = await fetch(`${this.backendUrl}/api/v1/query-selected`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: question,
          selected_text: selectedText,
          page_context: pageContext
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in selected text query:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.backendUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

// Create a global instance
const apiService = new ApiService();

export default apiService;