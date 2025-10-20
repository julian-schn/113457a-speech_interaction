import { defineConfig } from "vitepress";

export default defineConfig({
  title: "113457a Speech Interaction",
  description: "Documentation and notes for 113457a Speech Interaction",
  base: "/113457a-speech_interaction/",

  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Moodle", link: "https://moodle.hdm-stuttgart.de/course/view.php?id=23405" },
      { text: "Assignment", link: "/assignment" },
      { text: "Notes", link: "https://heisler.pages.mi.hdm-stuttgart.de/si/intro.html"},
      { text: "Repo", link: "https://github.com/julian-schn/113457a-speech_interaction"},
    ],

    sidebar: {
      "/": [
        {
          text: "Documentation",
          items: [{ text: "Assignment", link: "/assignment" }],
        },
        {
          text: "Notes Archive",
          collapsed: false,
          items: [
            {
              text: "2025-10-21 Notes",
              link: "/notes-archive/2025-10-21 notes",
            },
          ],
        },
      ],
    },

    search: {
      provider: "local",
    },

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/julian-schn/113457a-speech_interaction",
      },
    ],
  },
});
