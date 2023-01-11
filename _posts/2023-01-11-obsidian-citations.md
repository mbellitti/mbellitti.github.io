---
title:  "Handling citations with Obsidian+Zotero"
excerpt_separator: "<!--more-->"
date: 2023-01-11
categories:
  - Blog
---

<!-- <script type="text/javascript" async -->
  <!-- src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> -->
<!-- </script> -->

All my notes are organized in [Obsidian](https://obsidian.md/) vaults. Doing research for my PhD I needed a way to keep track of the papers I read, and cite them appropriately. My approach is based on two components:
- Zotero
- The Obsidian [Citation](https://github.com/hans/obsidian-citation-plugin) plugin

[Zotero](https://www.zotero.org/) is a library manager: it creates and maintains a database in which each entry is a primary source --for example, a paper-- with associated metadata (title, authors, journal, date, etc.). It's a very powerful tool, but for this post we only need a couple of functions. 

Once we have some sources in our library, acquired using one of Zotero's [connectors](https://www.zotero.org/download/connectors), the library should look something like this

![Zotero Library](/assets/images/zotero_library.png)

at this point we have to make the library accessible from Obsidian. The Obsidian citation plugin is compatible with the BibTeX format, so we install the [better BibTeX](https://github.com/retorquere/zotero-better-bibtex) plugin for Zotero, and export our entire library to a `.bib` file. Make sure to select "keep updated" when exporting, so that whenever we add a new source to Zotero the `.bib` file will also be updated.

Right click on "My Library" > select "Export Library" > Choose "Better BibTeX" format > save the file. I saved it as `complete_literature.bib`.

![Zotero Export settings](/assets/images/zotero_export.png)

Now, open Obsidian settings and configure the Citations plugins to use the file we just created as Citations database. We can also pick a directory where all our literature notes will be kept.

![Obsidian citations settings](/assets/images/obsidian_citations.png)

Under "Literature note content template" paste this:
```
{% raw %}
{{title}}
*{{authorString}}* ({{year}})

{{#if entry.files}}[[{{entry.files.[0]}}|Open PDF]]{{/if}} [See in Zotero]({{zoteroSelectURI}}) [Resolve DOI](https://doi.org/{{DOI}})
{% endraw %}
```
and for "Literature note title template" use `{% raw %}@{{citekey}}{% endraw %}`

And we're done. The resulting workflow is this:
- Find a paper on the internet
- Add it to Zotero
- The library gets automatically exported, so the paper is now available in Obsdian
- Cite it: writing a note in Obsidian, open the command palette (Ctrl+P) and select "Insert literature note link". 

![Obsidian citation](/assets/images/obsidian_citation_menu.png)


This creates a note in your vault that looks like this:

![Obsidian literature note](/assets/images/obsidian_note_example.png)

Using the Zotero citation key preceded by a @ as the note title has two advantages:
1. It's really easy to see at a glance how your literature notes are connected in the graph view (filter by '@' in the title)
2. It plays nicely when exporting to markdown, but that's another story.

Here is an example of one of my notes that includes some citations: 

![Citations example](/assets/images/citations_example.png)

Hovering any of those link opens a preview with the corresopnding literature note, and from that I can open the PDF, resolve the DOI, or highlight the entry in Zotero. That's all, I hope this will make your research easier.