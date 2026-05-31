% Minimal AIAA-style class for conference papers
% Based on the official AIAA LaTeX template structure
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{new-aiaa}[2024/01/01 Minimal AIAA Conference Paper]

\newif\if@conf
\newif\if@journal
\DeclareOption{conf}{\@conftrue}
\DeclareOption{journal}{\@journaltrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[11pt]{article}

% Page layout
\RequirePackage[margin=1in]{geometry}
\RequirePackage{setspace}
\setstretch{1.0}

% Fonts
\RequirePackage[T1]{fontenc}
\RequirePackage{newtxtext,newtxmath}

% Standard packages
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{booktabs}
\RequirePackage{float}
\RequirePackage{enumitem}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{natbib}
\setcitestyle{numbers,square,comma}
\bibliographystyle{unsrtnat}

% Title formatting
\RequirePackage{titlesec}
\titleformat{\section}{\normalfont\bfseries\large}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalfont\bfseries\normalsize}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalfont\itshape\normalsize}{\thesubsubsection.}{0.5em}{}

% Title page
\renewcommand{\maketitle}{%
  \begingroup
  \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
  \begin{center}%
    {\LARGE\bfseries \@title \par}%
    \vskip 1.5em%
    {\large \@author \par}%
  \end{center}%
  \par\vskip 1.5em%
  \endgroup
  \setcounter{footnote}{0}%
}

% Abstract
\renewenvironment{abstract}{%
  \begin{center}%
    \bfseries Abstract%
  \end{center}%
  \begin{quotation}\noindent
}{%
  \end{quotation}%
}

% Nomenclature environment
\newenvironment{nomenclature}{%
  \section*{Nomenclature}%
  \begin{description}[labelwidth=2cm,leftmargin=2.5cm,style=sameline]%
}{%
  \end{description}%
}

\endinput
