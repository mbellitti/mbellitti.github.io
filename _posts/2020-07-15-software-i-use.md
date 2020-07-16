--- 
title: "Some software I use" 
excerpt_separator: "<!--more-->" 
date: 2020-07-15 
toc: true 
categories:
  - Blog 
tags:
  - software 
---

This is an incomplete list of the software (and some hardware) I use daily.

I love vi keybindings. 

# OS and Desktop 
My main machine runs Ubuntu 20.04 with i3wm. 

Ubuntu is a popular distro, with excellent hardware support. I don't like their obsession with snaps, so I will probably have to migrate away soon.

i3wm is pure window management bliss. I configured it to have vi-like keybindings.

# Text Editing 
I use [Neovim](https://neovim.io/). It feels like vim, but fixes some odd quirks that I could never quite get to work reliably (mostly colorscheme issues). 

My favorite colorscheme is
[vim-monochrome](https://github.com/fxn/vim-monochrome). The visual mode
highlight color is beautiful.

# Web Browsing
My main browser is Firefox. I also like to use [qutebrowser](https://qutebrowser.org/) (vi keybindings!), when I need a quick
answer.

The two most important addons I use are [Imagus](https://addons.mozilla.org/en-US/firefox/addon/imagus/) and [uBlock Origin](https://addons.mozilla.org/en-US/firefox/addon/ublock-origin/).  

I also have a [pi-hole](https://pi-hole.net/) set up at home, but mostly because I had a pi zero that was sitting unused in a drawer.

# Note Taking 
I don't have a general purpose note system I find natural, but for science related matters my current workflow is:

1. Write rough handwritten notes using a Wacom intuos S (it works fine, but a slightly larger one would be more comfortable) and
[Xournal++](https://github.com/xournalpp/xournalpp)
2. Once I understand enough the topic I type up some clean notes using Neovim+vimtex+[zathura](https://pwmt.org/projects/zathura/) (zathura is a simple pdf viewer with vi keybindings.). This combination provides forward and backward search from the LaTeX source to the pdf. 

Two nice things about zathura is the possibility to quickly turn a pdf to "dark mode" using `:set recolor` and the quick access to the pdf table of content pressing the Tab key.
