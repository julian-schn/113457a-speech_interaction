import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '113457a Speech Interaction',
  description: 'Documentation and notes for 113457a Speech Interaction',
  base: '/113457a-speech_interaction/',
  
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/first_doc' }
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'First Document', link: '/first_doc' }
        ]
      }
    ],

    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/julian-schn/113457a-speech_interaction' }
    ]
  }
})