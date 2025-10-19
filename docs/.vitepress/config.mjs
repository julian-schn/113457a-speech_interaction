import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Hello World 23',
  description: 'Documentation site',
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

    socialLinks: [
      { icon: 'github', link: 'https://github.com/julian-schn/113457a-speech_interaction' }
    ]
  }
})