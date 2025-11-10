import { defineConfig } from 'vitepress';
import { withSidebar } from 'vitepress-sidebar';

// Wrap your VitePress options with vitepress-sidebar to auto-generate the sidebar
export default defineConfig(
  withSidebar(
    {
      title: '113457a Speech Interaction',
      description: 'Documentation and notes for 113457a Speech Interaction',
      base: '/113457a-speech_interaction/',

      themeConfig: {
        // --- NAV (still manually curated) ---
        nav: [
          { text: 'Home', link: '/' },
          {
            text: 'Moodle',
            link: 'https://moodle.hdm-stuttgart.de/course/view.php?id=23405'
          },
          { text: 'Assignment', link: '/assignment' },
          {
            text: 'Notes',
            items: [
              {
                text: 'Lecture Slides',
                link: 'https://heisler.pages.mi.hdm-stuttgart.de/si/intro.html'
              },
              { text: 'Project Notes Archive', link: '/notes-archive/' }
            ]
          },
          {
            text: 'Repo',
            link: 'https://github.com/julian-schn/113457a-speech_interaction'
          }
        ],

        // Search + social links (kept from your config)
        search: { provider: 'local' },
        socialLinks: [
          {
            icon: 'github',
            link: 'https://github.com/julian-schn/113457a-speech_interaction'
          }
        ]
      }
    },

    // --- vitepress-sidebar options (auto sidebar) ---
    {
      // Project root -> the docs folder that contains .vitepress and your markdown
      documentRootPath: '/docs',

      // Nice defaults; tweak as you prefer
      collapsed: false,           // show groups expanded by default
      collapseDepth: 2,           // how deep auto groups can collapse
      capitalizeFirst: true,      // prettify file/folder names
      hyphenToSpace: true,        // turn "my-file" into "my file"
      underscoreToSpace: true,

      // Examples of useful filters you can uncomment/add:
      // excludeByGlobPattern: ['README.md', 'node_modules/', 'public/'],
      // includeRootIndexFile: true,
      // includeFolderIndexFile: true
    }
  )
);
