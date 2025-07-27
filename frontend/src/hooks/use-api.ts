/**
 * React Query hooks for API interactions
 */

import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

// TODO: Implement React Query hooks when building features
// This file provides the structure for API state management

export function useSendMessage() {
  return useMutation({
    mutationFn: async ({ message }: { message: string }) => {
      return api.chat.send(message)
    },
    // TODO: Add error handling and cache invalidation
  })
}

export function useUploadDocument() {
  return useMutation({
    mutationFn: (file: File) => api.documents.upload(file),
    // TODO: Add progress tracking and error handling
  })
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => api.health.check(),
    // TODO: Configure refetch intervals and error handling
  })
}