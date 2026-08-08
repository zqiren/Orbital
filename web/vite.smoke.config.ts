// Temporary config for smoke-testing against the isolated dev daemon on
// port 8321 (the packaged Orbital.app owns 8000). Not used by any npm
// script — safe to delete after the beta-badge visual pass.
import { mergeConfig } from 'vitest/config'
import base from './vite.config'

export default mergeConfig(base, {
  server: {
    proxy: {
      '/api': 'http://localhost:8321',
      '/ws': {
        target: 'ws://localhost:8321',
        ws: true,
      },
    },
  },
})
