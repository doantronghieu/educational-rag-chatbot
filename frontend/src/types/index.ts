/**
 * Type definitions and Zod schemas for the RAG chatbot application
 */

import { z } from 'zod'

// TODO: These types will be auto-generated from Prisma schema
// For now, we define basic types for development

// Basic data types
export type Message = {
  id: string
  content: string
  role: 'USER' | 'ASSISTANT'
  createdAt: Date
}

export type Document = {
  id: string
  filename: string
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  createdAt: Date
}

// Zod schemas for validation
export const ChatMessageSchema = z.object({
  message: z.string().min(1, 'Message cannot be empty').max(2000, 'Message too long'),
})

export const FileUploadSchema = z.object({
  file: z.instanceof(File)
    .refine((file) => file.size <= 10 * 1024 * 1024, 'File must be less than 10MB')
    .refine((file) => file.type === 'application/pdf', 'Only PDF files allowed'),
})

// Inferred types
export type ChatMessage = z.infer<typeof ChatMessageSchema>
export type FileUpload = z.infer<typeof FileUploadSchema>

// API response types
export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  error?: string
}

// UI state types
export interface ChatState {
  messages: Message[]
  isLoading: boolean
  error: string | null
}