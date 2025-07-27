/**
 * API client for backend communication
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    }

    const response = await fetch(url, config)

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  // Chat endpoints
  chat = {
    send: async (message: string, sessionId?: string) => {
      return this.request('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message, session_id: sessionId }),
      })
    },

    getHistory: async (sessionId: string) => {
      return this.request(`/api/chat/history/${sessionId}`)
    },

    getSessions: async () => {
      return this.request('/api/chat/sessions')
    },

    createSession: async () => {
      return this.request('/api/chat/sessions', { method: 'POST' })
    },
  }

  // Document endpoints
  documents = {
    upload: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)

      return this.request('/api/documents/upload', {
        method: 'POST',
        headers: {}, // Remove Content-Type to let browser set it with boundary
        body: formData,
      })
    },

    list: async (page = 1, limit = 20) => {
      return this.request(`/api/documents?page=${page}&limit=${limit}`)
    },

    get: async (documentId: string) => {
      return this.request(`/api/documents/${documentId}`)
    },

    delete: async (documentId: string) => {
      return this.request(`/api/documents/${documentId}`, {
        method: 'DELETE',
      })
    },

    getStatus: async (documentId: string) => {
      return this.request(`/api/documents/${documentId}/status`)
    },
  }

  // Health check
  health = {
    check: async () => {
      return this.request('/api/health')
    },
  }
}

export const api = new ApiClient()
export default api