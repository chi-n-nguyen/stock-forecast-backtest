/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        'tc-bg':        '#0a0a0a',
        'tc-panel':     '#111111',
        'tc-surface':   '#1a1a1a',
        'tc-hover':     '#222222',
        'tc-border':    '#2a2a2a',
        'tc-border-hi': '#444444',
        'tc-grid':      '#1e1e1e',
        'tc-text':      '#e0e0e0',
        'tc-meta':      '#888888',
        'tc-dim':       '#555555',
        'tc-cyan':      '#00e5ff',
        'tc-green':     '#00e676',
        'tc-red':       '#ff1744',
        'tc-amber':     '#ffab00',
        'tc-purple':    '#b388ff',
      },
      fontFamily: {
        mono:    ["'IBM Plex Mono'", 'monospace'],
        display: ["'JetBrains Mono'", 'monospace'],
      },
    }
  },
  plugins: []
}
