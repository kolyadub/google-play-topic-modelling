




<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-BmnGTzITfSCD9SKlVfngZdzNq8Fa33lRq00rF1eRsg4zcCH3VtX8QtS6687+5GdeaVj1LzKyLj6+oXJLcswj6w==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-0669c64f32137d2083f522a555f9e065.css" />
  
    <link crossorigin="anonymous" media="all" integrity="sha512-0IGYt6nc+cz/zU4CV44Lb8IcwmWQlYmSAHsCzOtlyo8lLyvQz1jNirl1Ld8CbxalQ5fCiFufvxPm5RLVW+Mo4Q==" rel="stylesheet" href="https://github.githubassets.com/assets/github-d08198b7a9dcf9ccffcd4e02578e0b6f.css" />
    
    
    
    


  <meta name="viewport" content="width=device-width">
  
  <title>pace/pace.js at master · HubSpot/pace</title>
    <meta name="description" content="Automatically add a progress bar to your site. #hubspot-open-source - HubSpot/pace">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    <meta name="twitter:image:src" content="https://avatars1.githubusercontent.com/u/326419?s=400&amp;v=4" /><meta name="twitter:site" content="@github" /><meta name="twitter:card" content="summary" /><meta name="twitter:title" content="HubSpot/pace" /><meta name="twitter:description" content="Automatically add a progress bar to your site. #hubspot-open-source - HubSpot/pace" />
    <meta property="og:image" content="https://avatars1.githubusercontent.com/u/326419?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="HubSpot/pace" /><meta property="og:url" content="https://github.com/HubSpot/pace" /><meta property="og:description" content="Automatically add a progress bar to your site. #hubspot-open-source - HubSpot/pace" />

  <link rel="assets" href="https://github.githubassets.com/">
    <link rel="web-socket" href="wss://live.github.com/_sockets/VjI6NTI3MzgzODIzOmI1NDY2M2Y1MDIwY2QyZDM1NmE5OWMzYmM5Njk1ZTFlNTM5MjM2NDNhYjAyNTljZjQ5MzgxYzg1YzBjMDYxMDM=--a917b55010306ab880dde03a2cad674ef6d44bd7">
  <link rel="sudo-modal" href="/sessions/sudo_modal">

  <meta name="request-id" content="E1E2:429EF:1434D7:1C99AC:5EE20419" data-pjax-transient="true" /><meta name="html-safe-nonce" content="4fe185b4841ad6f0a8e2ec28fca451688f668e99" data-pjax-transient="true" /><meta name="visitor-payload" content="eyJyZWZlcnJlciI6Imh0dHBzOi8vZ2l0aHViLmNvbS9IdWJTcG90L3BhY2UiLCJyZXF1ZXN0X2lkIjoiRTFFMjo0MjlFRjoxNDM0RDc6MUM5OUFDOjVFRTIwNDE5IiwidmlzaXRvcl9pZCI6IjIyMjAyMzkzMjE4Mzk5MDM4NTQiLCJyZWdpb25fZWRnZSI6ImFtcyIsInJlZ2lvbl9yZW5kZXIiOiJpYWQifQ==" data-pjax-transient="true" /><meta name="visitor-hmac" content="6d4399df340ce2bcc95f8180e74cc83635866537d4284e57fe0f88ac0e12cdf4" data-pjax-transient="true" />



  <meta name="github-keyboard-shortcuts" content="repository,source-code" data-pjax-transient="true" />

  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="c1kuD-K2HIVF635lypcsWPoD4kilo5-jA_wBFyT4uMY">
  <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

<meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-ga_id" content="" class="js-octo-ga-id" /><meta name="octolytics-actor-id" content="44906229" /><meta name="octolytics-actor-login" content="kolyadub" /><meta name="octolytics-actor-hash" content="c162be8bdb92120ca60d4c6946e87957c84a3c7d66c80d99c4b7ebe380dd9a0b" />
<meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />


<meta name="optimizely-sdk-key" content="cowimJNste4j7QnBNCjaw" />

    <meta name="google-analytics" content="UA-3769691-2">

  <meta class="js-ga-set" name="userId" content="69f36a19a870ce603a8002a01d016618">

<meta class="js-ga-set" name="dimension10" content="Responsive">

<meta class="js-ga-set" name="dimension1" content="Logged In">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="kolyadub">


      <meta name="expected-hostname" content="github.com">

      <meta name="js-proxy-site-detection-payload" content="NWQwYWM5MjhiM2ZkY2FiZDBhNzA5ZmY3OTI4MDU1MjdjNzJiNzZlOWFiYmQwOTY4YjI5MzBlMDMxNWY3MDYyNXx7InJlbW90ZV9hZGRyZXNzIjoiMjEzLjIxLjYwLjQ4IiwicmVxdWVzdF9pZCI6IkUxRTI6NDI5RUY6MTQzNEQ3OjFDOTlBQzo1RUUyMDQxOSIsInRpbWVzdGFtcCI6MTU5MTg3MDQ5NSwiaG9zdCI6ImdpdGh1Yi5jb20ifQ==">

    <meta name="enabled-features" content="MARKETPLACE_PENDING_INSTALLATIONS,PAGE_STALE_CHECK">

  <meta http-equiv="x-pjax-version" content="a8cb16a75e33142dcb26db12a795343e">
  

      <link href="https://github.com/HubSpot/pace/commits/master.atom" rel="alternate" title="Recent Commits to pace:master" type="application/atom+xml">

  <meta name="go-import" content="github.com/HubSpot/pace git https://github.com/HubSpot/pace.git">

  <meta name="octolytics-dimension-user_id" content="326419" /><meta name="octolytics-dimension-user_login" content="HubSpot" /><meta name="octolytics-dimension-repository_id" content="12774825" /><meta name="octolytics-dimension-repository_nwo" content="HubSpot/pace" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="12774825" /><meta name="octolytics-dimension-repository_network_root_nwo" content="HubSpot/pace" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/HubSpot/pace/blob/master/pace.js" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="alternate icon" class="js-site-favicon" type="image/png" href="https://github.githubassets.com/favicons/favicon.png">
  <link rel="icon" class="js-site-favicon" type="image/svg+xml" href="https://github.githubassets.com/favicons/favicon.svg">

<meta name="theme-color" content="#1e2327">


  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-in env-production page-responsive page-blob">
    

    <div class="position-relative js-header-wrapper ">
      <a href="#start-of-content" class="p-3 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
      <span class="Progress progress-pjax-loader position-fixed width-full js-pjax-loader-bar">
        <span class="progress-pjax-loader-bar top-0 left-0" style="width: 0%;"></span>
      </span>

      
      



          <header class="Header py-lg-0 js-details-container Details flex-wrap flex-lg-nowrap p-responsive" role="banner">
  <div class="Header-item d-none d-lg-flex">
    <a class="Header-link" href="https://github.com/" data-hotkey="g d"
  aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>

  </div>

  <div class="Header-item d-lg-none">
    <button class="Header-link btn-link js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
      <svg height="24" class="octicon octicon-three-bars" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M1 2.75A.75.75 0 011.75 2h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 2.75zm0 5A.75.75 0 011.75 7h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 7.75zM1.75 12a.75.75 0 100 1.5h12.5a.75.75 0 100-1.5H1.75z"></path></svg>
    </button>
  </div>

  <div class="Header-item Header-item--full flex-column flex-lg-row width-full flex-order-2 flex-lg-order-none mr-0 mr-lg-3 mt-3 mt-lg-0 Details-content--hidden">
        <div class="header-search header-search-current js-header-search-current  flex-self-stretch flex-lg-self-auto mr-0 mr-lg-3 mb-3 mb-lg-0 scoped-search site-scoped-search js-site-search position-relative js-jump-to js-header-search-current-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" role="search" aria-label="Site" data-scope-type="Repository" data-scope-id="12774825" data-scoped-search-url="/HubSpot/pace/search" data-unscoped-search-url="/search" action="/HubSpot/pace/search" accept-charset="UTF-8" method="get">
      <label class="form-control input-sm header-search-wrapper p-0 header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control input-sm header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search or jump to…"
          data-unscoped-placeholder="Search or jump to…"
          data-scoped-placeholder="Search or jump to…"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search or jump to…"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations"
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" value="jR5zaH/OFEokLswl5M+SxeOKetw9lNOSnEj43ejvIpsd9u3MqGyGDlUhw3df2MdoWji0nxCL89/bX6WiYiCCFQ==" data-csrf="true" class="js-data-jump-to-suggestions-path-csrf" />
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


    <li class="d-flex flex-justify-center flex-items-center p-0 f5 js-jump-to-suggestion">
      <img src="https://github.githubassets.com/images/spinners/octocat-spinner-128.gif" alt="Octocat Spinner Icon" class="m-2" width="28">
    </li>
</ul>

            </div>
      </label>
</form>  </div>
</div>


    <nav class="d-flex flex-column flex-lg-row flex-self-stretch flex-lg-self-auto" aria-label="Global">
    <a class="Header-link py-lg-3 d-block d-lg-none py-2 border-top border-lg-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:dashboard:user" aria-label="Dashboard" href="/dashboard">
      Dashboard
</a>
  <a class="js-selected-navigation-item Header-link py-lg-3  mr-0 mr-lg-3 py-2 border-top border-lg-top-0 border-white-fade-15" data-hotkey="g p" data-ga-click="Header, click, Nav menu - item:pulls context:user" aria-label="Pull requests you created" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls" href="/pulls">
    Pull requests
</a>
  <a class="js-selected-navigation-item Header-link py-lg-3  mr-0 mr-lg-3 py-2 border-top border-lg-top-0 border-white-fade-15" data-hotkey="g i" data-ga-click="Header, click, Nav menu - item:issues context:user" aria-label="Issues you created" data-selected-links="/issues /issues/assigned /issues/mentioned /issues" href="/issues">
    Issues
</a>

    <div class="mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15">
      <a class="js-selected-navigation-item Header-link py-lg-3 d-inline-block" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-octo-click="marketplace_click" data-octo-dimensions="location:nav_bar" data-selected-links=" /marketplace" href="/marketplace">
        Marketplace
</a>      

    </div>

  <a class="js-selected-navigation-item Header-link py-lg-3  mr-0 mr-lg-3 py-2 border-top border-lg-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore" href="/explore">
    Explore
</a>


    <a class="Header-link d-block d-lg-none mr-0 mr-lg-3 py-2 py-lg-3 border-top border-lg-top-0 border-white-fade-15" href="/kolyadub">
      <img class="avatar avatar-user" src="https://avatars3.githubusercontent.com/u/44906229?s=40&amp;v=4" width="20" height="20" alt="@kolyadub" />
      kolyadub
</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form action="/logout" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="ra09vltF35xlceXxpbNeT1+UtonvV84l7V+xhE4VHVzawqwL2iJ8W2716p1SuAOADgJyxUhKKvmC5EIRmI0NdA==" />
      <button type="submit" class="Header-link mr-0 mr-lg-3 py-2 py-lg-3 border-top border-lg-top-0 border-white-fade-15 d-lg-none btn-link d-block width-full text-left" data-ga-click="Header, sign out, icon:logout" style="padding-left: 2px;">
        <svg class="octicon octicon-sign-out v-align-middle" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.75C2 1.784 2.784 1 3.75 1h2.5a.75.75 0 010 1.5h-2.5a.25.25 0 00-.25.25v10.5c0 .138.112.25.25.25h2.5a.75.75 0 010 1.5h-2.5A1.75 1.75 0 012 13.25V2.75zm10.44 4.5H6.75a.75.75 0 000 1.5h5.69l-1.97 1.97a.75.75 0 101.06 1.06l3.25-3.25a.75.75 0 000-1.06l-3.25-3.25a.75.75 0 10-1.06 1.06l1.97 1.97z"></path></svg>
        Sign out
      </button>
</form></nav>

  </div>

  <div class="Header-item Header-item--full flex-justify-center d-lg-none position-relative">
    <div class="css-truncate css-truncate-target width-fit position-absolute left-0 right-0 text-center">
                <svg class="octicon octicon-repo" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
    <a class="Header-link" href="/HubSpot">HubSpot</a>
    /
    <a class="Header-link" href="/HubSpot/pace">pace</a>

</div>
  </div>

  <div class="Header-item mr-0 mr-lg-3 flex-order-1 flex-lg-order-none">
    
    <a aria-label="You have no unread notifications" class="Header-link notification-indicator position-relative tooltipped tooltipped-sw js-socket-channel js-notification-indicator" data-hotkey="g n" data-ga-click="Header, go to notifications, icon:read" data-channel="notification-changed:44906229" href="/notifications">
        <span class="js-indicator-modifier mail-status "></span>
        <svg class="octicon octicon-bell" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 16a2 2 0 001.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 008 16z"></path><path fill-rule="evenodd" d="M8 1.5A3.5 3.5 0 004.5 5v2.947c0 .346-.102.683-.294.97l-1.703 2.556a.018.018 0 00-.003.01l.001.006c0 .002.002.004.004.006a.017.017 0 00.006.004l.007.001h10.964l.007-.001a.016.016 0 00.006-.004.016.016 0 00.004-.006l.001-.007a.017.017 0 00-.003-.01l-1.703-2.554a1.75 1.75 0 01-.294-.97V5A3.5 3.5 0 008 1.5zM3 5a5 5 0 0110 0v2.947c0 .05.015.098.042.139l1.703 2.555A1.518 1.518 0 0113.482 13H2.518a1.518 1.518 0 01-1.263-2.36l1.703-2.554A.25.25 0 003 7.947V5z"></path></svg>
</a>
  </div>


  <div class="Header-item position-relative d-none d-lg-flex">
    <details class="details-overlay details-reset">
  <summary class="Header-link"
      aria-label="Create new…"
      data-ga-click="Header, create new, icon:add">
    <svg class="octicon octicon-plus" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2a.75.75 0 01.75.75v4.5h4.5a.75.75 0 010 1.5h-4.5v4.5a.75.75 0 01-1.5 0v-4.5h-4.5a.75.75 0 010-1.5h4.5v-4.5A.75.75 0 018 2z"></path></svg> <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw mt-n2">
    
<a role="menuitem" class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a role="menuitem" class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a role="menuitem" class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a role="menuitem" class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>


  <div role="none" class="dropdown-divider"></div>
  <div class="dropdown-header">
    <span title="HubSpot/pace">This repository</span>
  </div>
    <a role="menuitem" class="dropdown-item" href="/HubSpot/pace/issues/new/choose" data-ga-click="Header, create new issue" data-skip-pjax>
      New issue
    </a>


  </details-menu>
</details>

  </div>

  <div class="Header-item position-relative mr-0 d-none d-lg-flex">
    
  <details class="details-overlay details-reset js-feature-preview-indicator-container" data-feature-preview-indicator-src="/users/kolyadub/feature_preview/indicator_check">

  <summary class="Header-link"
    aria-label="View profile and more"
    data-ga-click="Header, show menu, icon:avatar">
    <img
  alt="@kolyadub"
  width="20"
  height="20"
  src="https://avatars0.githubusercontent.com/u/44906229?s=60&amp;v=4"
  class="avatar avatar-user " />

      <span class="feature-preview-indicator js-feature-preview-indicator" style="top: 10px;" hidden></span>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw mt-n2" style="width: 180px">
    <div class="header-nav-current-user css-truncate"><a role="menuitem" class="no-underline user-profile-link px-3 pt-2 pb-2 mb-n2 mt-n1 d-block" href="/kolyadub" data-ga-click="Header, go to profile, text:Signed in as">Signed in as <strong class="css-truncate-target">kolyadub</strong></a></div>
    <div role="none" class="dropdown-divider"></div>

      <div class="pl-3 pr-3 f6 user-status-container js-user-status-context pb-1" data-url="/users/status?compact=1&amp;link_mentions=0&amp;truncate=1">
        
<div class="js-user-status-container
    user-status-compact rounded-1 px-2 py-1 mt-2
    border
  " data-team-hovercards-enabled>
  <details class="js-user-status-details details-reset details-overlay details-overlay-dark">
    <summary class="btn-link btn-block link-gray no-underline js-toggle-user-status-edit toggle-user-status-edit "
      role="menuitem" data-hydro-click="{&quot;event_type&quot;:&quot;user_profile.click&quot;,&quot;payload&quot;:{&quot;profile_user_id&quot;:326419,&quot;target&quot;:&quot;EDIT_USER_STATUS&quot;,&quot;user_id&quot;:44906229,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;}}" data-hydro-click-hmac="93e9926e0fd7f82adf009ecabbd00880100b0164f49483e9ab5d66626fba6d30">
      <div class="d-flex">
        <div class="f6 lh-condensed user-status-header
          d-inline-block v-align-middle
            user-status-emoji-only-header circle
            pr-2
"
            style="max-width: 29px"
          >
          <div class="user-status-emoji-container flex-shrink-0 mr-1 mt-1 lh-condensed-ultra v-align-bottom" style="">
            <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM5 8a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zM5.32 9.636a.75.75 0 011.038.175l.007.009c.103.118.22.222.35.31.264.178.683.37 1.285.37.602 0 1.02-.192 1.285-.371.13-.088.247-.192.35-.31l.007-.008a.75.75 0 111.222.87l-.614-.431c.614.43.614.431.613.431v.001l-.001.002-.002.003-.005.007-.014.019a1.984 1.984 0 01-.184.213c-.16.166-.338.316-.53.445-.63.418-1.37.638-2.127.629-.946 0-1.652-.308-2.126-.63a3.32 3.32 0 01-.715-.657l-.014-.02-.005-.006-.002-.003v-.002h-.001l.613-.432-.614.43a.75.75 0 01.183-1.044h.001z"></path></svg>
          </div>
        </div>
        <div class="
          d-inline-block v-align-middle
          
          
           css-truncate css-truncate-target 
           user-status-message-wrapper f6"
           style="line-height: 20px;" >
          <div class="d-inline-block text-gray-dark v-align-text-top text-left">
              <span class="text-gray ml-2">Set status</span>
          </div>
        </div>
      </div>
    </summary>
    <details-dialog class="details-dialog rounded-1 anim-fade-in fast Box Box--overlay" role="dialog" tabindex="-1">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="position-relative flex-auto js-user-status-form" action="/users/status?compact=1&amp;link_mentions=0&amp;truncate=1" accept-charset="UTF-8" method="post"><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="aVnmA/mSRa7BgLhczSbKRBlDJvivYGzIxJxeW1d7pygK4O+YGtjuJvnHYNt1oVEm2GSE0LMpc+x041d2cvH2aw==" />
        <div class="Box-header bg-gray border-bottom p-3">
          <button class="Box-btn-octicon js-toggle-user-status-edit btn-octicon float-right" type="reset" aria-label="Close dialog" data-close-dialog>
            <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
          </button>
          <h3 class="Box-title f5 text-bold text-gray-dark">Edit status</h3>
        </div>
        <input type="hidden" name="emoji" class="js-user-status-emoji-field" value="">
        <input type="hidden" name="organization_id" class="js-user-status-org-id-field" value="">
        <div class="px-3 py-2 text-gray-dark">
          <div class="js-characters-remaining-container position-relative mt-2">
            <div class="input-group d-table form-group my-0 js-user-status-form-group">
              <span class="input-group-button d-table-cell v-align-middle" style="width: 1%">
                <button type="button" aria-label="Choose an emoji" class="btn-outline btn js-toggle-user-status-emoji-picker btn-open-emoji-picker p-0">
                  <span class="js-user-status-original-emoji" hidden></span>
                  <span class="js-user-status-custom-emoji"></span>
                  <span class="js-user-status-no-emoji-icon" >
                    <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM5 8a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zM5.32 9.636a.75.75 0 011.038.175l.007.009c.103.118.22.222.35.31.264.178.683.37 1.285.37.602 0 1.02-.192 1.285-.371.13-.088.247-.192.35-.31l.007-.008a.75.75 0 111.222.87l-.614-.431c.614.43.614.431.613.431v.001l-.001.002-.002.003-.005.007-.014.019a1.984 1.984 0 01-.184.213c-.16.166-.338.316-.53.445-.63.418-1.37.638-2.127.629-.946 0-1.652-.308-2.126-.63a3.32 3.32 0 01-.715-.657l-.014-.02-.005-.006-.002-.003v-.002h-.001l.613-.432-.614.43a.75.75 0 01.183-1.044h.001z"></path></svg>
                  </span>
                </button>
              </span>
              <text-expander keys=": @" data-mention-url="/autocomplete/user-suggestions" data-emoji-url="/autocomplete/emoji">
                <input
                  type="text"
                  autocomplete="off"
                  data-no-org-url="/autocomplete/user-suggestions"
                  data-org-url="/suggestions?mention_suggester=1"
                  data-maxlength="80"
                  class="d-table-cell width-full form-control js-user-status-message-field js-characters-remaining-field"
                  placeholder="What's happening?"
                  name="message"
                  value=""
                  aria-label="What is your current status?">
              </text-expander>
              <div class="error">Could not update your status, please try again.</div>
            </div>
            <div style="margin-left: 53px" class="my-1 text-small label-characters-remaining js-characters-remaining" data-suffix="remaining" hidden>
              80 remaining
            </div>
          </div>
          <include-fragment class="js-user-status-emoji-picker" data-url="/users/status/emoji"></include-fragment>
          <div class="overflow-auto ml-n3 mr-n3 px-3 border-bottom" style="max-height: 33vh">
            <div class="user-status-suggestions js-user-status-suggestions collapsed overflow-hidden">
              <h4 class="f6 text-normal my-3">Suggestions:</h4>
              <div class="mx-3 mt-2 clearfix">
                  <div class="float-left col-6">
                      <button type="button" value=":palm_tree:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="palm_tree" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f334.png">🌴</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          On vacation
                        </div>
                      </button>
                      <button type="button" value=":face_with_thermometer:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="face_with_thermometer" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f912.png">🤒</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Out sick
                        </div>
                      </button>
                  </div>
                  <div class="float-left col-6">
                      <button type="button" value=":house:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="house" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3e0.png">🏠</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Working from home
                        </div>
                      </button>
                      <button type="button" value=":dart:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="dart" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3af.png">🎯</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Focusing
                        </div>
                      </button>
                  </div>
              </div>
            </div>
            <div class="user-status-limited-availability-container">
              <div class="form-checkbox my-0">
                <input type="checkbox" name="limited_availability" value="1" class="js-user-status-limited-availability-checkbox" data-default-message="I may be slow to respond." aria-describedby="limited-availability-help-text-truncate-true-compact-true" id="limited-availability-truncate-true-compact-true">
                <label class="d-block f5 text-gray-dark mb-1" for="limited-availability-truncate-true-compact-true">
                  Busy
                </label>
                <p class="note" id="limited-availability-help-text-truncate-true-compact-true">
                  When others mention you, assign you, or request your review,
                  GitHub will let them know that you have limited availability.
                </p>
              </div>
            </div>
          </div>
          <div class="d-inline-block f5 mr-2 pt-3 pb-2" >
  <div class="d-inline-block mr-1">
    Clear status
  </div>

  <details class="js-user-status-expire-drop-down f6 dropdown details-reset details-overlay d-inline-block mr-2">
    <summary class="f5 btn-link link-gray-dark border px-2 py-1 rounded-1" aria-haspopup="true">
      <div class="js-user-status-expiration-interval-selected d-inline-block v-align-baseline">
        Never
      </div>
      <div class="dropdown-caret"></div>
    </summary>

    <ul class="dropdown-menu dropdown-menu-se pl-0 overflow-auto" style="width: 220px; max-height: 15.5em">
      <li>
        <button type="button" class="btn-link dropdown-item js-user-status-expire-button ws-normal" title="Never">
          <span class="d-inline-block text-bold mb-1">Never</span>
          <div class="f6 lh-condensed">Keep this status until you clear your status or edit your status.</div>
        </button>
      </li>
      <li class="dropdown-divider" role="none"></li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 30 minutes" value="2020-06-11T13:44:55+03:00">
            in 30 minutes
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 1 hour" value="2020-06-11T14:14:55+03:00">
            in 1 hour
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 4 hours" value="2020-06-11T17:14:55+03:00">
            in 4 hours
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="today" value="2020-06-11T23:59:59+03:00">
            today
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="this week" value="2020-06-14T23:59:59+03:00">
            this week
          </button>
        </li>
    </ul>
  </details>
  <input class="js-user-status-expiration-date-input" type="hidden" name="expires_at" value="">
</div>

          <include-fragment class="js-user-status-org-picker" data-url="/users/status/organizations"></include-fragment>
        </div>
        <div class="d-flex flex-items-center flex-justify-between p-3 border-top">
          <button type="submit" disabled class="width-full btn btn-primary mr-2 js-user-status-submit">
            Set status
          </button>
          <button type="button" disabled class="width-full js-clear-user-status-button btn ml-2 ">
            Clear status
          </button>
        </div>
</form>    </details-dialog>
  </details>
</div>

      </div>
      <div role="none" class="dropdown-divider"></div>

    <a role="menuitem" class="dropdown-item" href="/kolyadub" data-ga-click="Header, go to profile, text:your profile">Your profile</a>

    <a role="menuitem" class="dropdown-item" href="/kolyadub?tab=repositories" data-ga-click="Header, go to repositories, text:your repositories">Your repositories</a>


    <a role="menuitem" class="dropdown-item" href="/kolyadub?tab=projects" data-ga-click="Header, go to projects, text:your projects">Your projects</a>

    <a role="menuitem" class="dropdown-item" href="/kolyadub?tab=stars" data-ga-click="Header, go to starred repos, text:your stars">Your stars</a>
      <a role="menuitem" class="dropdown-item" href="https://gist.github.com/mine" data-ga-click="Header, your gists, text:your gists">Your gists</a>





    <div role="none" class="dropdown-divider"></div>
      <a role="menuitem" class="dropdown-item" href="/settings/billing" data-ga-click="Header, go to billing, text:upgrade" >Upgrade</a>
      
<div id="feature-enrollment-toggle" class="hide-sm hide-md feature-preview-details position-relative">
  <button
    type="button"
    class="dropdown-item btn-link"
    role="menuitem"
    data-feature-preview-trigger-url="/users/kolyadub/feature_previews"
    data-feature-preview-close-details="{&quot;event_type&quot;:&quot;feature_preview.clicks.close_modal&quot;,&quot;payload&quot;:{&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}"
    data-feature-preview-close-hmac="b9fa92d0f18e396690ea5a84079782df26bef227001fd9a27c46aac368b5880f"
    data-hydro-click="{&quot;event_type&quot;:&quot;feature_preview.clicks.open_modal&quot;,&quot;payload&quot;:{&quot;link_location&quot;:&quot;user_dropdown&quot;,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}"
    data-hydro-click-hmac="0a66afd96f05e506d24d20ce224d2f00403c4a425b5a54ebc6216e819d445da2"
  >
    Feature preview
  </button>
    <span class="feature-preview-indicator js-feature-preview-indicator" hidden></span>
</div>

    <a role="menuitem" class="dropdown-item" href="https://help.github.com" data-ga-click="Header, go to help, text:help">Help</a>
    <a role="menuitem" class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings">Settings</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="logout-form" action="/logout" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="e2QZbrfVAkvjR29uV95NhL/O/YAPuP87nrvR9SFMiAQMC4jbNrKhjOjDYAKg1RBL7lg5zKilG+fxACJg99SYLA==" />
      
      <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout" role="menuitem">
        Sign out
      </button>
      <input type="text" name="required_field_5cbe" hidden="hidden" class="form-control" /><input type="hidden" name="timestamp" value="1591870495844" class="form-control" /><input type="hidden" name="timestamp_secret" value="5cdabcc2c76ed0b6288acd4a407c8550ef163db00b8b536f57238a10ef59c3f7" class="form-control" />
</form>  </details-menu>
</details>

  </div>

</header>

        

    </div>

  <div id="start-of-content" class="show-on-focus"></div>




    <div id="js-flash-container">


  <template class="js-flash-template">
    <div class="flash flash-full  js-flash-template-container">
  <div class="container-lg px-2" >
    <button class="flash-close js-flash-close" type="button" aria-label="Dismiss this message">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    
      <div class="js-flash-template-message"></div>

  </div>
</div>
  </template>
</div>


      

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>




  <div class="application-main " data-commit-hovercards-enabled>
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main  >
      

  




  










  <div class="pagehead repohead hx_repohead readability-menu bg-gray-light pb-0 pt-0 pt-lg-3">

    <div class="d-flex container-lg mb-4 p-responsive d-none d-lg-flex">

      <div class="flex-auto min-width-0 width-fit mr-3">
        <h1 class="public  d-flex flex-wrap flex-items-center break-word float-none ">
  <span class="flex-self-stretch" style="margin-top: -2px;">
      <svg class="octicon octicon-repo" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
  </span>
  <span class="author ml-2 flex-self-stretch" itemprop="author">
    <a class="url fn" rel="author" data-hovercard-type="organization" data-hovercard-url="/orgs/HubSpot/hovercard" href="/HubSpot">HubSpot</a>
  </span>
  <span class="path-divider flex-self-stretch">/</span>
  <strong itemprop="name" class="mr-2 flex-self-stretch">
    <a data-pjax="#js-repo-pjax-container" href="/HubSpot/pace">pace</a>
  </strong>
  
</h1>


      </div>

      <ul class="pagehead-actions flex-shrink-0 " >



    <li >
      
    <details class="dropdown details-reset details-overlay d-inline-block float-left"
      data-deferred-details-content-url="/HubSpot/pace/used_by_contents"
    >
      <summary class="btn btn-sm btn-with-count" data-menu-button>
        <svg class="octicon octicon-package v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.878.392a1.75 1.75 0 00-1.756 0l-5.25 3.045A1.75 1.75 0 001 4.951v6.098c0 .624.332 1.2.872 1.514l5.25 3.045a1.75 1.75 0 001.756 0l5.25-3.045c.54-.313.872-.89.872-1.514V4.951c0-.624-.332-1.2-.872-1.514L8.878.392zM7.875 1.69a.25.25 0 01.25 0l4.63 2.685L8 7.133 3.245 4.375l4.63-2.685zM2.5 5.677v5.372c0 .09.047.171.125.216l4.625 2.683V8.432L2.5 5.677zm6.25 8.271l4.625-2.683a.25.25 0 00.125-.216V5.677L8.75 8.432v5.516z"></path></svg>
        Used by
        <div class="dropdown-caret"></div>
      </summary>
      <include-fragment accept="text/fragment+html">
        <div class="dropdown-menu dropdown-menu-s p-3 text-center" style="width:360px;">
          <img width="32" height="32" alt="Loading..." class="my-0" src="https://github.githubassets.com/images/spinners/octocat-spinner-64.gif" />
          <p class="pt-1 m-0 f5 text-gray-light">
            Loading dependents...
          </p>
        </div>
      </include-fragment>
    </details>
    <a class="social-count"
      href="/HubSpot/pace/network/dependents?package_id=UGFja2FnZS00OTMzNzA3NzY%3D"
      aria-label="3389 repositories depend on this package"
    >
      3.4k
    </a>

    </li>

  <li>
    

    <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" class="js-social-form js-social-container clearfix" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="tG791iZQ8OlLfJ1BvO9ngIICnyo6IrQFprPk5kdCrweweK7cM+iTl55IYEZAtCQ37LsCxtW74tHpePWjTL81dg==" />      <input type="hidden" name="repository_id" value="12774825">

      <details class="details-reset details-overlay select-menu float-left" >
        <summary class="select-menu-button float-left btn btn-sm btn-with-count" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;WATCH_BUTTON&quot;,&quot;repository_id&quot;:12774825,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}" data-hydro-click-hmac="6a70df5ad5ff95532e6aa1512abfab4ba4b24bd7b9d3adeebb6a519b9bcf4e60" data-ga-click="Repository, click Watch settings, action:blob#show">          <span data-menu-button>
              <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
              Watch
          </span>
</summary>        <details-menu
          class="select-menu-modal position-absolute mt-5 "
          style="z-index: 99;">
          <div class="select-menu-header">
            <span class="select-menu-title">Notifications</span>
          </div>
          <div class="select-menu-list">
            <button
              type="submit"
              name="do"
              value="included"
              class="select-menu-item width-full"
              aria-checked="true"
              role="menuitemradio"
              
            >
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Not watching</span>
                <span class="description">Be notified only when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Watch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="release_only" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Releases only</span>
                <span class="description">Be notified of new releases, and when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Unwatch releases
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="subscribed" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Watching</span>
                <span class="description">Be notified of all conversations.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Unwatch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="ignore" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Ignoring</span>
                <span class="description">Never be notified.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-bell-slash" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1.5c-.997 0-1.895.416-2.534 1.086A.75.75 0 014.38 1.55 5 5 0 0113 5v2.373a.75.75 0 01-1.5 0V5A3.5 3.5 0 008 1.5zM4.182 4.31L1.19 2.143a.75.75 0 10-.88 1.214L3 5.305v2.642a.25.25 0 01-.042.139L1.255 10.64A1.518 1.518 0 002.518 13h11.108l1.184.857a.75.75 0 10.88-1.214l-1.375-.996a1.196 1.196 0 00-.013-.01L4.198 4.321a.733.733 0 00-.016-.011zm7.373 7.19L4.5 6.391v1.556c0 .346-.102.683-.294.97l-1.703 2.556a.018.018 0 00-.003.01.015.015 0 00.005.012.017.017 0 00.006.004l.007.001h9.037zM8 16a2 2 0 001.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 008 16z"></path></svg>
                  Stop ignoring
                </span>
              </div>
            </button>
          </div>
        </details-menu>
      </details>
        <a class="social-count js-social-count"
          href="/HubSpot/pace/watchers"
          aria-label="441 users are watching this repository">
          441
        </a>
</form>
  </li>

  <li>
      <div class="js-toggler-container js-social-container starring-container ">
    <form class="starred js-social-form" action="/HubSpot/pace/unstar" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="yPomtHlInPMMMstp3EE6wjUgjWCR5NpQRTBFGlVgMGH7ex0yEH43WDfhZW+FZMUL5wlaZLR2BFhy+FrecxxNcw==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count  js-toggler-target" aria-label="Unstar this repository" title="Unstar HubSpot/pace" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;UNSTAR_BUTTON&quot;,&quot;repository_id&quot;:12774825,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}" data-hydro-click-hmac="7a20d14f6dbf1fec8514c96991d076d4a4faf12da5efb2bc89827f4da92eaa98" data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">        <svg height="16" class="octicon octicon-star-fill" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"></path></svg>

        Unstar
</button>        <a class="social-count js-social-count" href="/HubSpot/pace/stargazers"
           aria-label="14872 users starred this repository">
           14.9k
        </a>
</form>
    <form class="unstarred js-social-form" action="/HubSpot/pace/star" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="mwaTV32OjdvUU/fFZhJteocLoOqbaLKNbQfH0n3TQ3QOKFXTM5Y+wqkYSHUOboI1ktANqiFZKNsKh9kV90rTJw==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count  js-toggler-target" aria-label="Unstar this repository" title="Star HubSpot/pace" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;STAR_BUTTON&quot;,&quot;repository_id&quot;:12774825,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}" data-hydro-click-hmac="6819a7cd9c1ecfbb2fc9055323136f522411a208bfcb76d7d4c4db5d6c87eb71" data-ga-click="Repository, click star button, action:blob#show; text:Star">        <svg height="16" class="octicon octicon-star" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path></svg>

        Star
</button>        <a class="social-count js-social-count" href="/HubSpot/pace/stargazers"
           aria-label="14872 users starred this repository">
          14.9k
        </a>
</form>  </div>

  </li>

  <li>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="btn-with-count" action="/HubSpot/pace/fork" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="37LJzC1gnT2/2ZMzEDGPZl4RDUJa6WqNhM5s/ldOUtAghY9PzS9I58ESZ8adgUQLCi9LTTwhb7v3nXwzqEoR5g==" />
            <button class="btn btn-sm btn-with-count" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;FORK_BUTTON&quot;,&quot;repository_id&quot;:12774825,&quot;originating_url&quot;:&quot;https://github.com/HubSpot/pace/blob/master/pace.js&quot;,&quot;user_id&quot;:44906229}}" data-hydro-click-hmac="5041230f93ec9a73c8ae5d83d909cc302b9cf023bfa5673b84deca41770d7523" data-ga-click="Repository, show fork modal, action:blob#show; text:Fork" type="submit" title="Fork your own copy of HubSpot/pace to your account" aria-label="Fork your own copy of HubSpot/pace to your account">              <svg class="octicon octicon-repo-forked" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path></svg>
              Fork
</button></form>
    <a href="/HubSpot/pace/network/members" class="social-count"
       aria-label="1933 users forked this repository">
      1.9k
    </a>
  </li>
</ul>

    </div>
      <nav class="js-repo-nav js-sidenav-container-pjax clearfix hx_reponav reponav p-responsive d-none d-lg-block container-lg"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
    aria-label="Repository"
     data-pjax="#js-repo-pjax-container">
  <ul class="list-style-none">
    <li  itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a class="js-selected-navigation-item selected reponav-item" itemprop="url" data-hotkey="g c" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /HubSpot/pace" href="/HubSpot/pace">
        <div class="d-inline"><svg class="octicon octicon-code" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path></svg></div>
        <span itemprop="name">Code</span>
        <meta itemprop="position" content="1">
</a>    </li>

      <li  itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" data-hotkey="g i" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /HubSpot/pace/issues" href="/HubSpot/pace/issues">
          <div class="d-inline"><svg class="octicon octicon-issue-opened" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM0 8a8 8 0 1116 0A8 8 0 010 8zm9 3a1 1 0 11-2 0 1 1 0 012 0zm-.25-6.25a.75.75 0 00-1.5 0v3.5a.75.75 0 001.5 0v-3.5z"></path></svg></div>
          <span itemprop="name">Issues</span>
          <span class="Counter">279</span>
          <meta itemprop="position" content="2">
</a>      </li>

    <li  itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a data-hotkey="g p" data-skip-pjax="true" itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /HubSpot/pace/pulls" href="/HubSpot/pace/pulls">
        <div class="d-inline"><svg class="octicon octicon-git-pull-request" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z"></path></svg></div>
        <span itemprop="name">Pull requests</span>
        <span class="Counter">23</span>
        <meta itemprop="position" content="4">
</a>    </li>


      <li itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement" class="position-relative float-left ">
        <a data-hotkey="g w" data-skip-pjax="true" class="js-selected-navigation-item reponav-item" data-selected-links="repo_actions /HubSpot/pace/actions" href="/HubSpot/pace/actions">
          <div class="d-inline"><svg class="octicon octicon-play" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM6.379 5.227A.25.25 0 006 5.442v5.117a.25.25 0 00.379.214l4.264-2.559a.25.25 0 000-.428L6.379 5.227z"></path></svg></div>
          Actions
</a>
      </li>

      <li >
        <a data-hotkey="g b" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /HubSpot/pace/projects" href="/HubSpot/pace/projects">
          <div class="d-inline"><svg class="octicon octicon-project" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg></div>
          Projects
          <span class="Counter">0</span>
</a>      </li>

      <li >
        <a class="js-selected-navigation-item reponav-item" data-hotkey="g w" data-selected-links="repo_wiki /HubSpot/pace/wiki" href="/HubSpot/pace/wiki">
          <div class="d-inline"><svg class="octicon octicon-book" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm8.755 3a2.25 2.25 0 012.25-2.25H14.5v9h-3.757c-.71 0-1.4.201-1.992.572l.004-7.322zm-1.504 7.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574z"></path></svg></div>
          Wiki
</a>      </li>

      <li >
        <a data-skip-pjax="true" class="js-selected-navigation-item reponav-item" data-selected-links="security overview alerts policy token_scanning code_scanning /HubSpot/pace/security" href="/HubSpot/pace/security">
          <div class="d-inline"><svg class="octicon octicon-shield" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.467.133a1.75 1.75 0 011.066 0l5.25 1.68A1.75 1.75 0 0115 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.7 1.7 0 01-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 011.217-1.667l5.25-1.68zm.61 1.429a.25.25 0 00-.153 0l-5.25 1.68a.25.25 0 00-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.2.2 0 00.154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.25.25 0 00-.174-.237l-5.25-1.68zM9 10.5a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.75a.75.75 0 10-1.5 0v3a.75.75 0 001.5 0v-3z"></path></svg></div>
          Security
              <span class="Counter js-security-tab-count" data-url="/HubSpot/pace/security/overall-count" hidden></span>
</a>      </li>

      <li >
        <a class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people /HubSpot/pace/pulse" href="/HubSpot/pace/pulse">
          <div class="d-inline"><svg class="octicon octicon-graph" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 1.75a.75.75 0 00-1.5 0v12.5c0 .414.336.75.75.75h14.5a.75.75 0 000-1.5H1.5V1.75zm14.28 2.53a.75.75 0 00-1.06-1.06L10 7.94 7.53 5.47a.75.75 0 00-1.06 0L3.22 8.72a.75.75 0 001.06 1.06L7 7.06l2.47 2.47a.75.75 0 001.06 0l5.25-5.25z"></path></svg></div>
          Insights
</a>      </li>


  </ul>
</nav>

  <div class="reponav-wrapper reponav-small d-lg-none">
  <nav class="reponav js-reponav text-center no-wrap"
       itemscope
       itemtype="http://schema.org/BreadcrumbList">

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a class="js-selected-navigation-item selected reponav-item" itemprop="url" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /HubSpot/pace" href="/HubSpot/pace">
        <span itemprop="name">Code</span>
        <meta itemprop="position" content="1">
</a>    </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /HubSpot/pace/issues" href="/HubSpot/pace/issues">
          <span itemprop="name">Issues</span>
          <span class="Counter">279</span>
          <meta itemprop="position" content="2">
</a>      </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /HubSpot/pace/pulls" href="/HubSpot/pace/pulls">
        <span itemprop="name">Pull requests</span>
        <span class="Counter">23</span>
        <meta itemprop="position" content="4">
</a>    </span>


      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /HubSpot/pace/projects" href="/HubSpot/pace/projects">
          <span itemprop="name">Projects</span>
          <span class="Counter">0</span>
          <meta itemprop="position" content="5">
</a>      </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_actions /HubSpot/pace/actions" href="/HubSpot/pace/actions">
          <span itemprop="name">Actions</span>
          <meta itemprop="position" content="6">
</a>      </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_wiki /HubSpot/pace/wiki" href="/HubSpot/pace/wiki">
          <span itemprop="name">Wiki</span>
          <meta itemprop="position" content="7">
</a>      </span>

      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="security overview alerts policy token_scanning code_scanning /HubSpot/pace/security" href="/HubSpot/pace/security">
        <span itemprop="name">Security</span>
            <span class="Counter js-security-deferred-tab-count" hidden></span>
        <meta itemprop="position" content="8">
</a>
      <a class="js-selected-navigation-item reponav-item" data-selected-links="pulse /HubSpot/pace/pulse" href="/HubSpot/pace/pulse">
        Pulse
</a>
      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="community /HubSpot/pace/community" href="/HubSpot/pace/community">
          Community
</a>      </span>

  </nav>
</div>


  </div>

  

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>


<div class="container-lg clearfix new-discussion-timeline  p-responsive">
  <div class="repository-content ">

    
    

  


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/HubSpot/pace/blob/4afddfb37aec45cc8588a91416fe28ca173314a8/pace.js">Permalink</a>

    <!-- blob contrib key: blob_contributors:v22:9b0c934e6d780762285821fbe6e97cf6 -->
    

    <div class="d-flex flex-items-start flex-shrink-0 flex-column flex-md-row pb-3">
      <span class="d-flex flex-justify-between width-full width-md-auto">
        
<details class="details-reset details-overlay branch-select-menu " id="branch-select-menu">
  <summary class="btn css-truncate btn-sm"
           data-hotkey="w"
           title="Switch branches or tags">
    <i>Branch:</i>
    <span class="css-truncate-target" data-menu-button>master</span>
    <span class="dropdown-caret"></span>
  </summary>

  <details-menu class="SelectMenu SelectMenu--hasFilter" src="/HubSpot/pace/refs/master/pace.js?source_action=show&amp;source_controller=blob" preload>
    <div class="SelectMenu-modal">
      <include-fragment class="SelectMenu-loading" aria-label="Menu is loading">
        <svg class="octicon octicon-octoface anim-pulse" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M14.7 5.34c.13-.32.55-1.59-.13-3.31 0 0-1.05-.33-3.44 1.3-1-.28-2.07-.32-3.13-.32s-2.13.04-3.13.32c-2.39-1.64-3.44-1.3-3.44-1.3-.68 1.72-.26 2.99-.13 3.31C.49 6.21 0 7.33 0 8.69 0 13.84 3.33 15 7.98 15S16 13.84 16 8.69c0-1.36-.49-2.48-1.3-3.35zM8 14.02c-3.3 0-5.98-.15-5.98-3.35 0-.76.38-1.48 1.02-2.07 1.07-.98 2.9-.46 4.96-.46 2.07 0 3.88-.52 4.96.46.65.59 1.02 1.3 1.02 2.07 0 3.19-2.68 3.35-5.98 3.35zM5.49 9.01c-.66 0-1.2.8-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.54-1.78-1.2-1.78zm5.02 0c-.66 0-1.2.79-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.53-1.78-1.2-1.78z"></path></svg>
      </include-fragment>
    </div>
  </details-menu>
</details>

        <div class="BtnGroup flex-shrink-0 d-md-none">
          <a href="/HubSpot/pace/find/master"
                class="js-pjax-capture-input btn btn-sm BtnGroup-item"
                data-pjax
                data-hotkey="t">
            Find file
          </a>
          <clipboard-copy value="pace.js" class="btn btn-sm BtnGroup-item">
            Copy path
          </clipboard-copy>
        </div>
      </span>
      <h2 id="blob-path" class="breadcrumb flex-auto min-width-0 text-normal flex-md-self-center ml-md-2 mr-md-3 my-2 my-md-0">
        <span class="js-repo-root text-bold"><span class="js-path-segment d-inline-block wb-break-all"><a data-pjax="true" href="/HubSpot/pace"><span>pace</span></a></span></span><span class="separator">/</span><strong class="final-path">pace.js</strong>
      </h2>

      <div class="BtnGroup flex-shrink-0 d-none d-md-inline-block">
        <a href="/HubSpot/pace/find/master"
              class="js-pjax-capture-input btn btn-sm BtnGroup-item"
              data-pjax
              data-hotkey="t">
          Find file
        </a>
        <clipboard-copy value="pace.js" class="btn btn-sm BtnGroup-item">
          Copy path
        </clipboard-copy>
      </div>
    </div>



    
  <div class="Box Box--condensed d-flex flex-column flex-shrink-0 mb-3">
      <div class="Box-body d-flex flex-justify-between bg-blue-light flex-column flex-md-row flex-items-start flex-md-items-center">
        <span class="pr-md-4 f6">
          <a rel="contributor" data-skip-pjax="true" data-hovercard-type="user" data-hovercard-url="/users/Friss/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/Friss"><img class="avatar avatar-user" src="https://avatars3.githubusercontent.com/u/4588318?s=40&amp;v=4" width="20" height="20" alt="@Friss" /></a>
          <a class="text-bold link-gray-dark lh-default v-align-middle" rel="contributor" data-hovercard-type="user" data-hovercard-url="/users/Friss/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/Friss">Friss</a>
            <span class="lh-default v-align-middle">
              <a data-pjax="true" title="Use ::" class="link-gray" href="/HubSpot/pace/commit/20a7a70028f51c5e511bcb98be5e116f577faa9f">Use ::</a>
            </span>
        </span>
        <span class="d-inline-block flex-shrink-0 v-align-bottom f6 mt-2 mt-md-0">
          <a class="pr-2 text-mono link-gray" href="/HubSpot/pace/commit/20a7a70028f51c5e511bcb98be5e116f577faa9f" data-pjax>20a7a70</a>
          <relative-time datetime="2014-11-06T00:30:49Z" class="no-wrap">Nov 6, 2014</relative-time>
        </span>
      </div>

    <div class="Box-body d-flex flex-items-center flex-auto f6 border-bottom-0 flex-wrap" >
      <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark float-left mr-2" id="blob_contributors_box">
        <summary class="btn-link">
          <span><strong>5</strong> contributors</span>
        </summary>
        <details-dialog
          class="Box Box--overlay d-flex flex-column anim-fade-in fast"
          aria-label="Users who have contributed to this file"
          src="/HubSpot/pace/contributors-list/master/pace.js" preload>
          <div class="Box-header">
            <button class="Box-btn-octicon btn-octicon float-right" type="button" aria-label="Close dialog" data-close-dialog>
              <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"></path></svg>
            </button>
            <h3 class="Box-title">
              Users who have contributed to this file
            </h3>
          </div>
          <include-fragment class="octocat-spinner my-3" aria-label="Loading..."></include-fragment>
        </details-dialog>
      </details>
        <span class="">
    <a class="avatar-link" data-hovercard-type="user" data-hovercard-url="/users/zackbloom/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/HubSpot/pace/commits/master/pace.js?author=zackbloom">
      <img class="avatar mr-1 avatar-user" src="https://avatars0.githubusercontent.com/u/55347?s=40&amp;v=4" width="20" height="20" alt="@zackbloom" /> 
</a>    <a class="avatar-link" data-hovercard-type="user" data-hovercard-url="/users/Friss/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/HubSpot/pace/commits/master/pace.js?author=Friss">
      <img class="avatar mr-1 avatar-user" src="https://avatars3.githubusercontent.com/u/4588318?s=40&amp;v=4" width="20" height="20" alt="@Friss" /> 
</a>    <a class="avatar-link" data-hovercard-type="user" data-hovercard-url="/users/adamschwartz/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/HubSpot/pace/commits/master/pace.js?author=adamschwartz">
      <img class="avatar mr-1 avatar-user" src="https://avatars3.githubusercontent.com/u/154613?s=40&amp;v=4" width="20" height="20" alt="@adamschwartz" /> 
</a>    <a class="avatar-link" data-hovercard-type="user" data-hovercard-url="/users/alexmelnikov/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/HubSpot/pace/commits/master/pace.js?author=alexmelnikov">
      <img class="avatar mr-1 avatar-user" src="https://avatars1.githubusercontent.com/u/614087?s=40&amp;v=4" width="20" height="20" alt="@alexmelnikov" /> 
</a>    <a class="avatar-link" data-hovercard-type="user" data-hovercard-url="/users/kaleworsley/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/HubSpot/pace/commits/master/pace.js?author=kaleworsley">
      <img class="avatar mr-1 avatar-user" src="https://avatars3.githubusercontent.com/u/164566?s=40&amp;v=4" width="20" height="20" alt="@kaleworsley" /> 
</a>
</span>

    </div>
  </div>






    <div class="Box mt-3 position-relative
      ">
      
<div class="Box-header py-2 d-flex flex-column flex-shrink-0 flex-md-row flex-md-items-center">
  <div class="text-mono f6 flex-auto pr-3 flex-order-2 flex-md-order-1 mt-2 mt-md-0">

      935 lines (832 sloc)
      <span class="file-info-divider"></span>
    25.9 KB
  </div>

  <div class="d-flex py-1 py-md-0 flex-auto flex-order-1 flex-md-order-2 flex-sm-grow-0 flex-justify-between">

    <div class="BtnGroup">
      <a id="raw-url" class="btn btn-sm BtnGroup-item" href="/HubSpot/pace/raw/master/pace.js">Raw</a>
        <a class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b" href="/HubSpot/pace/blame/master/pace.js">Blame</a>
      <a rel="nofollow" class="btn btn-sm BtnGroup-item" href="/HubSpot/pace/commits/master/pace.js">History</a>
    </div>


    <div>
          <a class="btn-octicon tooltipped tooltipped-nw js-remove-unless-platform"
             data-platforms="windows,mac"
             href="x-github-client://openRepo/https://github.com/HubSpot/pace?branch=master&amp;filepath=pace.js"
             aria-label="Open this file in GitHub Desktop"
             data-ga-click="Repository, open with desktop">
              <svg class="octicon octicon-device-desktop" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.75 2.5h12.5a.25.25 0 01.25.25v7.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25v-7.5a.25.25 0 01.25-.25zM14.25 1H1.75A1.75 1.75 0 000 2.75v7.5C0 11.216.784 12 1.75 12h3.727c-.1 1.041-.52 1.872-1.292 2.757A.75.75 0 004.75 16h6.5a.75.75 0 00.565-1.243c-.772-.885-1.193-1.716-1.292-2.757h3.727A1.75 1.75 0 0016 10.25v-7.5A1.75 1.75 0 0014.25 1zM9.018 12H6.982a5.72 5.72 0 01-.765 2.5h3.566a5.72 5.72 0 01-.765-2.5z"></path></svg>
          </a>

          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form js-update-url-with-hash" action="/HubSpot/pace/edit/master/pace.js" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="s0e3r39JZoUR/wtgcpbGzfUjrO1tW18YVbBFqmqDRINOGGEHna6jT0Lu1xTg5mgqmLTKeFG+mEjyj3xtu5K7LQ==" />
            <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
              aria-label="Edit the file in your fork of this project" data-hotkey="e" data-disable-with>
              <svg class="octicon octicon-pencil" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11.013 1.427a1.75 1.75 0 012.474 0l1.086 1.086a1.75 1.75 0 010 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 01-.927-.928l.929-3.25a1.75 1.75 0 01.445-.758l8.61-8.61zm1.414 1.06a.25.25 0 00-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 000-.354l-1.086-1.086zM11.189 6.25L9.75 4.81l-6.286 6.287a.25.25 0 00-.064.108l-.558 1.953 1.953-.558a.249.249 0 00.108-.064l6.286-6.286z"></path></svg>
            </button>
</form>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form" action="/HubSpot/pace/delete/master/pace.js" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="5M36gUTCkmKnjAukW+xZkMpi6f32Gz/pVnKlaPTV+kxS75d/nO6fpjqksvzVTm+KyCWlXvDL/GlhS33eF65XTw==" />
            <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
              aria-label="Delete the file in your fork of this project" data-disable-with>
              <svg class="octicon octicon-trashcan" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M6.5 1.75a.25.25 0 01.25-.25h2.5a.25.25 0 01.25.25V3h-3V1.75zm4.5 0V3h2.25a.75.75 0 010 1.5H2.75a.75.75 0 010-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75zM4.496 6.675a.75.75 0 10-1.492.15l.66 6.6A1.75 1.75 0 005.405 15h5.19c.9 0 1.652-.681 1.741-1.576l.66-6.6a.75.75 0 00-1.492-.149l-.66 6.6a.25.25 0 01-.249.225h-5.19a.25.25 0 01-.249-.225l-.66-6.6z"></path></svg>
            </button>
</form>    </div>
  </div>
</div>



      

  <div itemprop="text" class="Box-body p-0 blob-wrapper data type-javascript ">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8" data-paste-markdown-skip>
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>AjaxMonitor</span><span class=pl-kos>,</span> <span class=pl-v>Bar</span><span class=pl-kos>,</span> <span class=pl-v>DocumentMonitor</span><span class=pl-kos>,</span> <span class=pl-v>ElementMonitor</span><span class=pl-kos>,</span> <span class=pl-v>ElementTracker</span><span class=pl-kos>,</span> <span class=pl-v>EventLagMonitor</span><span class=pl-kos>,</span> <span class=pl-v>Evented</span><span class=pl-kos>,</span> <span class=pl-v>Events</span><span class=pl-kos>,</span> <span class=pl-v>NoTargetError</span><span class=pl-kos>,</span> <span class=pl-v>Pace</span><span class=pl-kos>,</span> <span class=pl-v>RequestIntercept</span><span class=pl-kos>,</span> <span class=pl-c1>SOURCE_KEYS</span><span class=pl-kos>,</span> <span class=pl-v>Scaler</span><span class=pl-kos>,</span> <span class=pl-v>SocketRequestTracker</span><span class=pl-kos>,</span> <span class=pl-v>XHRRequestTracker</span><span class=pl-kos>,</span> <span class=pl-s1>animation</span><span class=pl-kos>,</span> <span class=pl-s1>avgAmplitude</span><span class=pl-kos>,</span> <span class=pl-s1>bar</span><span class=pl-kos>,</span> <span class=pl-s1>cancelAnimation</span><span class=pl-kos>,</span> <span class=pl-s1>cancelAnimationFrame</span><span class=pl-kos>,</span> <span class=pl-s1>defaultOptions</span><span class=pl-kos>,</span> <span class=pl-s1>extend</span><span class=pl-kos>,</span> <span class=pl-s1>extendNative</span><span class=pl-kos>,</span> <span class=pl-s1>getFromDOM</span><span class=pl-kos>,</span> <span class=pl-s1>getIntercept</span><span class=pl-kos>,</span> <span class=pl-s1>handlePushState</span><span class=pl-kos>,</span> <span class=pl-s1>ignoreStack</span><span class=pl-kos>,</span> <span class=pl-s1>init</span><span class=pl-kos>,</span> <span class=pl-s1>now</span><span class=pl-kos>,</span> <span class=pl-s1>options</span><span class=pl-kos>,</span> <span class=pl-s1>requestAnimationFrame</span><span class=pl-kos>,</span> <span class=pl-s1>result</span><span class=pl-kos>,</span> <span class=pl-s1>runAnimation</span><span class=pl-kos>,</span> <span class=pl-s1>scalers</span><span class=pl-kos>,</span> <span class=pl-s1>shouldIgnoreURL</span><span class=pl-kos>,</span> <span class=pl-s1>shouldTrack</span><span class=pl-kos>,</span> <span class=pl-s1>source</span><span class=pl-kos>,</span> <span class=pl-s1>sources</span><span class=pl-kos>,</span> <span class=pl-s1>uniScaler</span><span class=pl-kos>,</span> <span class=pl-s1>_WebSocket</span><span class=pl-kos>,</span> <span class=pl-s1>_XDomainRequest</span><span class=pl-kos>,</span> <span class=pl-s1>_XMLHttpRequest</span><span class=pl-kos>,</span> <span class=pl-s1>_i</span><span class=pl-kos>,</span> <span class=pl-s1>_intercept</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span><span class=pl-kos>,</span> <span class=pl-s1>_pushState</span><span class=pl-kos>,</span> <span class=pl-s1>_ref</span><span class=pl-kos>,</span> <span class=pl-s1>_ref1</span><span class=pl-kos>,</span> <span class=pl-s1>_replaceState</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>__slice</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>slice</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>__hasProp</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>.</span><span class=pl-c1>hasOwnProperty</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>__extends</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>child</span><span class=pl-kos>,</span> <span class=pl-s1>parent</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>key</span> <span class=pl-k>in</span> <span class=pl-s1>parent</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>__hasProp</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>parent</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-s1>child</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>parent</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-kos>}</span> <span class=pl-k>function</span> <span class=pl-en>ctor</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>constructor</span> <span class=pl-c1>=</span> <span class=pl-s1>child</span><span class=pl-kos>;</span> <span class=pl-kos>}</span> <span class=pl-s1>ctor</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span> <span class=pl-c1>=</span> <span class=pl-s1>parent</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>;</span> <span class=pl-s1>child</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>ctor</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>child</span><span class=pl-kos>.</span><span class=pl-c1>__super__</span> <span class=pl-c1>=</span> <span class=pl-s1>parent</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>child</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>__indexOf</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>indexOf</span> <span class=pl-c1>||</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>item</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>l</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-k>in</span> <span class=pl-smi>this</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-smi>this</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> <span class=pl-s1>item</span><span class=pl-kos>)</span> <span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>;</span> <span class=pl-kos>}</span> <span class=pl-k>return</span> <span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>defaultOptions</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>catchupTime</span>: <span class=pl-c1>100</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>initialRate</span>: <span class=pl-c1>.03</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>minTime</span>: <span class=pl-c1>250</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>ghostTime</span>: <span class=pl-c1>100</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>maxProgressPerFrame</span>: <span class=pl-c1>20</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>easeFactor</span>: <span class=pl-c1>1.25</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>startOnPageLoad</span>: <span class=pl-c1>true</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>restartOnPushState</span>: <span class=pl-c1>true</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>restartOnRequestAfter</span>: <span class=pl-c1>500</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>target</span>: <span class=pl-s>&#39;body&#39;</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>elements</span>: <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>checkInterval</span>: <span class=pl-c1>100</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>selectors</span>: <span class=pl-kos>[</span><span class=pl-s>&#39;body&#39;</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>eventLag</span>: <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>minSamples</span>: <span class=pl-c1>10</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>sampleCount</span>: <span class=pl-c1>3</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>lagThreshold</span>: <span class=pl-c1>3</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>ajax</span>: <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>trackMethods</span>: <span class=pl-kos>[</span><span class=pl-s>&#39;GET&#39;</span><span class=pl-kos>]</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>trackWebSockets</span>: <span class=pl-c1>true</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>ignoreURLs</span>: <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>now</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>_ref</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-kos>(</span><span class=pl-s1>_ref</span> <span class=pl-c1>=</span> <span class=pl-k>typeof</span> <span class=pl-s1>performance</span> !== <span class=pl-s>&quot;undefined&quot;</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>performance</span> !== null ? <span class=pl-k>typeof</span> <span class=pl-s1>performance</span><span class=pl-kos>.</span><span class=pl-c1>now</span> <span class=pl-c1>===</span> <span class=pl-s>&quot;function&quot;</span> ? <span class=pl-s1>performance</span><span class=pl-kos>.</span><span class=pl-en>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref</span> : <span class=pl-c1>+</span><span class=pl-kos>(</span><span class=pl-k>new</span> <span class=pl-v>Date</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>requestAnimationFrame</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>requestAnimationFrame</span> <span class=pl-c1>||</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>mozRequestAnimationFrame</span> <span class=pl-c1>||</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>webkitRequestAnimationFrame</span> <span class=pl-c1>||</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>msRequestAnimationFrame</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>cancelAnimationFrame</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>cancelAnimationFrame</span> <span class=pl-c1>||</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>mozCancelAnimationFrame</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>requestAnimationFrame</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>requestAnimationFrame</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>fn</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-s1>fn</span><span class=pl-kos>,</span> <span class=pl-c1>50</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cancelAnimationFrame</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>id</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-en>clearTimeout</span><span class=pl-kos>(</span><span class=pl-s1>id</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>runAnimation</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>fn</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>last</span><span class=pl-kos>,</span> <span class=pl-s1>tick</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>last</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tick</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>diff</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>diff</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>last</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>diff</span> &gt;= <span class=pl-c1>33</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>last</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>fn</span><span class=pl-kos>(</span><span class=pl-s1>diff</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>requestAnimationFrame</span><span class=pl-kos>(</span><span class=pl-s1>tick</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-s1>tick</span><span class=pl-kos>,</span> <span class=pl-c1>33</span> <span class=pl-c1>-</span> <span class=pl-s1>diff</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>tick</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>result</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>args</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-s1>obj</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>obj</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>key</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>args</span> <span class=pl-c1>=</span> <span class=pl-c1>3</span> &lt;= <span class=pl-smi>arguments</span><span class=pl-kos>.</span><span class=pl-c1>length</span> ? <span class=pl-s1>__slice</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>arguments</span><span class=pl-kos>,</span> <span class=pl-c1>2</span><span class=pl-kos>)</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>obj</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;function&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>obj</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-s1>obj</span><span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>obj</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>extend</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-s1>out</span><span class=pl-kos>,</span> <span class=pl-s1>source</span><span class=pl-kos>,</span> <span class=pl-s1>sources</span><span class=pl-kos>,</span> <span class=pl-s1>val</span><span class=pl-kos>,</span> <span class=pl-s1>_i</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>out</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>sources</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span> &lt;= <span class=pl-smi>arguments</span><span class=pl-kos>.</span><span class=pl-c1>length</span> ? <span class=pl-s1>__slice</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>arguments</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span> <span class=pl-c1>=</span> <span class=pl-s1>sources</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>sources</span><span class=pl-kos>[</span><span class=pl-s1>_i</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>source</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>key</span> <span class=pl-k>in</span> <span class=pl-s1>source</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-s1>__hasProp</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>source</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-k>continue</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>val</span> <span class=pl-c1>=</span> <span class=pl-s1>source</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>out</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> != null<span class=pl-kos>)</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-k>typeof</span> <span class=pl-s1>out</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;object&#39;</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> != null<span class=pl-kos>)</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-k>typeof</span> <span class=pl-s1>val</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;object&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>extend</span><span class=pl-kos>(</span><span class=pl-s1>out</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>val</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>out</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>val</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>out</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>avgAmplitude</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>arr</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>count</span><span class=pl-kos>,</span> <span class=pl-s1>sum</span><span class=pl-kos>,</span> <span class=pl-s1>v</span><span class=pl-kos>,</span> <span class=pl-s1>_i</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>sum</span> <span class=pl-c1>=</span> <span class=pl-s1>count</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span> <span class=pl-c1>=</span> <span class=pl-s1>arr</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>v</span> <span class=pl-c1>=</span> <span class=pl-s1>arr</span><span class=pl-kos>[</span><span class=pl-s1>_i</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>sum</span> <span class=pl-c1>+=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>v</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>count</span><span class=pl-c1>++</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>sum</span> / <span class=pl-s1>count</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>getFromDOM</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-s1>json</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>data</span><span class=pl-kos>,</span> <span class=pl-s1>e</span><span class=pl-kos>,</span> <span class=pl-s1>el</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>key</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>key</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;options&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>json</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>json</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>el</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>querySelector</span><span class=pl-kos>(</span><span class=pl-s>&quot;[data-pace-&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>key</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;]&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-s1>el</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-en>getAttribute</span><span class=pl-kos>(</span><span class=pl-s>&quot;data-pace-&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>key</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-s1>json</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>data</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-c1>JSON</span><span class=pl-kos>.</span><span class=pl-en>parse</span><span class=pl-kos>(</span><span class=pl-s1>data</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>e</span> <span class=pl-c1>=</span> <span class=pl-s1>_error</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-k>typeof</span> <span class=pl-smi>console</span> !== <span class=pl-s>&quot;undefined&quot;</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-smi>console</span> !== null ? <span class=pl-smi>console</span><span class=pl-kos>.</span><span class=pl-en>error</span><span class=pl-kos>(</span><span class=pl-s>&quot;Error parsing inline pace options&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>e</span><span class=pl-kos>)</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Evented</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>Evented</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Evented</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>on</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span><span class=pl-kos>,</span> <span class=pl-s1>ctx</span><span class=pl-kos>,</span> <span class=pl-s1>once</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>_base</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>once</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>once</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_base</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_base</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">        <span class=pl-c1>handler</span>: <span class=pl-s1>handler</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">        <span class=pl-c1>ctx</span>: <span class=pl-s1>ctx</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">        <span class=pl-c1>once</span>: <span class=pl-s1>once</span></td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Evented</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>once</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span><span class=pl-kos>,</span> <span class=pl-s1>ctx</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span><span class=pl-kos>,</span> <span class=pl-s1>ctx</span><span class=pl-kos>,</span> <span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Evented</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>off</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-s1>_ref</span><span class=pl-kos>,</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_ref</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>handler</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-k>delete</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_results</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>while</span> <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>handler</span> <span class=pl-c1>===</span> <span class=pl-s1>handler</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>splice</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Evented</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>trigger</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>args</span><span class=pl-kos>,</span> <span class=pl-s1>ctx</span><span class=pl-kos>,</span> <span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span><span class=pl-kos>,</span> <span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-s1>once</span><span class=pl-kos>,</span> <span class=pl-s1>_ref</span><span class=pl-kos>,</span> <span class=pl-s1>_ref1</span><span class=pl-kos>,</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>event</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>args</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span> &lt;= <span class=pl-smi>arguments</span><span class=pl-kos>.</span><span class=pl-c1>length</span> ? <span class=pl-s1>__slice</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>arguments</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_ref</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_results</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>while</span> <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_ref1</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>handler</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref1</span><span class=pl-kos>.</span><span class=pl-c1>handler</span><span class=pl-kos>,</span> <span class=pl-s1>ctx</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref1</span><span class=pl-kos>.</span><span class=pl-c1>ctx</span><span class=pl-kos>,</span> <span class=pl-s1>once</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref1</span><span class=pl-kos>.</span><span class=pl-c1>once</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>handler</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-s1>ctx</span> != null ? <span class=pl-s1>ctx</span> : <span class=pl-smi>this</span><span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>once</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>event</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>splice</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Evented</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>Pace</span> <span class=pl-c1>||</span> <span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">  <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>Pace</span> <span class=pl-c1>=</span> <span class=pl-v>Pace</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>extend</span><span class=pl-kos>(</span><span class=pl-v>Pace</span><span class=pl-kos>,</span> <span class=pl-v>Evented</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>options</span> <span class=pl-c1>=</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>options</span> <span class=pl-c1>=</span> <span class=pl-s1>extend</span><span class=pl-kos>(</span><span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-s1>defaultOptions</span><span class=pl-kos>,</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>paceOptions</span><span class=pl-kos>,</span> <span class=pl-s1>getFromDOM</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>_ref</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s>&#39;ajax&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;document&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;eventLag&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;elements&#39;</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len</span><span class=pl-kos>;</span> <span class=pl-s1>_i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref</span><span class=pl-kos>[</span><span class=pl-s1>_i</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>[</span><span class=pl-s1>source</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> <span class=pl-c1>true</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>options</span><span class=pl-kos>[</span><span class=pl-s1>source</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>defaultOptions</span><span class=pl-kos>[</span><span class=pl-s1>source</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>NoTargetError</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_super</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>__extends</span><span class=pl-kos>(</span><span class=pl-v>NoTargetError</span><span class=pl-kos>,</span> <span class=pl-s1>_super</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>NoTargetError</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_ref1</span> <span class=pl-c1>=</span> <span class=pl-v>NoTargetError</span><span class=pl-kos>.</span><span class=pl-c1>__super__</span><span class=pl-kos>.</span><span class=pl-c1>constructor</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>_ref1</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>NoTargetError</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-v>Error</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Bar</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>Bar</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>getElement</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>targetElement</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>targetElement</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>querySelector</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>target</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-s1>targetElement</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>throw</span> <span class=pl-k>new</span> <span class=pl-v>NoTargetError</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>createElement</span><span class=pl-kos>(</span><span class=pl-s>&#39;div&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;pace pace-active&quot;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span><span class=pl-kos>.</span><span class=pl-en>replace</span><span class=pl-kos>(</span><span class=pl-pds>/pace-done/g</span><span class=pl-kos>,</span> <span class=pl-s>&#39;&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>+=</span> <span class=pl-s>&#39; pace-running&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span><span class=pl-kos>.</span><span class=pl-c1>innerHTML</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;&lt;div class=&quot;pace-progress&quot;&gt;\n  &lt;div class=&quot;pace-progress-inner&quot;&gt;&lt;/div&gt;\n&lt;/div&gt;\n&lt;div class=&quot;pace-activity&quot;&gt;&lt;/div&gt;&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>targetElement</span><span class=pl-kos>.</span><span class=pl-c1>firstChild</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>targetElement</span><span class=pl-kos>.</span><span class=pl-en>insertBefore</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span><span class=pl-kos>,</span> <span class=pl-s1>targetElement</span><span class=pl-kos>.</span><span class=pl-c1>firstChild</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>targetElement</span><span class=pl-kos>.</span><span class=pl-en>appendChild</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>finish</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>el</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>el</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>getElement</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>=</span> <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>className</span><span class=pl-kos>.</span><span class=pl-en>replace</span><span class=pl-kos>(</span><span class=pl-s>&#39;pace-active&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>+=</span> <span class=pl-s>&#39; pace-inactive&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span><span class=pl-kos>.</span><span class=pl-en>replace</span><span class=pl-kos>(</span><span class=pl-s>&#39;pace-running&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>body</span><span class=pl-kos>.</span><span class=pl-c1>className</span> <span class=pl-c1>+=</span> <span class=pl-s>&#39; pace-done&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>update</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>prog</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-s1>prog</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>render</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>destroy</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>getElement</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-en>removeChild</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>getElement</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>NoTargetError</span> <span class=pl-c1>=</span> <span class=pl-s1>_error</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>el</span> <span class=pl-c1>=</span> <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>render</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>el</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-s1>progressStr</span><span class=pl-kos>,</span> <span class=pl-s1>transform</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>querySelector</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>target</span><span class=pl-kos>)</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>el</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>getElement</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>transform</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;translate3d(&quot;</span> <span class=pl-c1>+</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;%, 0, 0)&quot;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s>&#39;webkitTransform&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;msTransform&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;transform&#39;</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>key</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>children</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>style</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>transform</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastRenderedProgress</span> <span class=pl-c1>||</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastRenderedProgress</span> | <span class=pl-c1>0</span> !== <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> | <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>children</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>setAttribute</span><span class=pl-kos>(</span><span class=pl-s>&#39;data-progress-text&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> | <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;%&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> &gt;= <span class=pl-c1>100</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>progressStr</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;99&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>progressStr</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>10</span> ? <span class=pl-s>&quot;0&quot;</span> : <span class=pl-s>&quot;&quot;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>progressStr</span> <span class=pl-c1>+=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> | <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>el</span><span class=pl-kos>.</span><span class=pl-c1>children</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>setAttribute</span><span class=pl-kos>(</span><span class=pl-s>&#39;data-progress&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>progressStr</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastRenderedProgress</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Bar</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>done</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> &gt;= <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Bar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Events</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>Events</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Events</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>trigger</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>name</span><span class=pl-kos>,</span> <span class=pl-s1>val</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>binding</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>name</span><span class=pl-kos>]</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>name</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_results</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>binding</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>binding</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>,</span> <span class=pl-s1>val</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Events</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>on</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>name</span><span class=pl-kos>,</span> <span class=pl-s1>fn</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>_base</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_base</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-s1>name</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_base</span><span class=pl-kos>[</span><span class=pl-s1>name</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>bindings</span><span class=pl-kos>[</span><span class=pl-s1>name</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>fn</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Events</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>_XMLHttpRequest</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>XMLHttpRequest</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>_XDomainRequest</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>XDomainRequest</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>_WebSocket</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>WebSocket</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>extendNative</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>to</span><span class=pl-kos>,</span> <span class=pl-s1>from</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>e</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_results</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>key</span> <span class=pl-k>in</span> <span class=pl-s1>from</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>to</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-k>typeof</span> <span class=pl-s1>from</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> !== <span class=pl-s>&#39;function&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-v>Object</span><span class=pl-kos>.</span><span class=pl-c1>defineProperty</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;function&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-v>Object</span><span class=pl-kos>.</span><span class=pl-en>defineProperty</span><span class=pl-kos>(</span><span class=pl-s1>to</span><span class=pl-kos>,</span> <span class=pl-s1>key</span><span class=pl-kos>,</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line">              <span class=pl-en>get</span>: <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line">                <span class=pl-k>return</span> <span class=pl-s1>from</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>}</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>configurable</span>: <span class=pl-c1>true</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>enumerable</span>: <span class=pl-c1>true</span></td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>to</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>from</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>[</span><span class=pl-s1>key</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>e</span> <span class=pl-c1>=</span> <span class=pl-s1>_error</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>ignoreStack</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>ignore</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>args</span><span class=pl-kos>,</span> <span class=pl-s1>fn</span><span class=pl-kos>,</span> <span class=pl-s1>ret</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fn</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>args</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span> &lt;= <span class=pl-smi>arguments</span><span class=pl-kos>.</span><span class=pl-c1>length</span> ? <span class=pl-s1>__slice</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>arguments</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ignoreStack</span><span class=pl-kos>.</span><span class=pl-en>unshift</span><span class=pl-kos>(</span><span class=pl-s>&#39;ignore&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ret</span> <span class=pl-c1>=</span> <span class=pl-s1>fn</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ignoreStack</span><span class=pl-kos>.</span><span class=pl-en>shift</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>ret</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>track</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>args</span><span class=pl-kos>,</span> <span class=pl-s1>fn</span><span class=pl-kos>,</span> <span class=pl-s1>ret</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fn</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-s1>args</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span> &lt;= <span class=pl-smi>arguments</span><span class=pl-kos>.</span><span class=pl-c1>length</span> ? <span class=pl-s1>__slice</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>arguments</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ignoreStack</span><span class=pl-kos>.</span><span class=pl-en>unshift</span><span class=pl-kos>(</span><span class=pl-s>&#39;track&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ret</span> <span class=pl-c1>=</span> <span class=pl-s1>fn</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ignoreStack</span><span class=pl-kos>.</span><span class=pl-en>shift</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>ret</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>shouldTrack</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>method</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>_ref2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>method</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>method</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;GET&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>ignoreStack</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;track&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s>&#39;force&#39;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-s1>ignoreStack</span><span class=pl-kos>.</span><span class=pl-c1>length</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ajax</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>method</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;socket&#39;</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ajax</span><span class=pl-kos>.</span><span class=pl-c1>trackWebSockets</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-s1>method</span><span class=pl-kos>.</span><span class=pl-en>toUpperCase</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-s1>__indexOf</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ajax</span><span class=pl-kos>.</span><span class=pl-c1>trackMethods</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>)</span> &gt;= <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>RequestIntercept</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_super</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>__extends</span><span class=pl-kos>(</span><span class=pl-v>RequestIntercept</span><span class=pl-kos>,</span> <span class=pl-s1>_super</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>RequestIntercept</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>monitorXHR</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line">      <span class=pl-v>RequestIntercept</span><span class=pl-kos>.</span><span class=pl-c1>__super__</span><span class=pl-kos>.</span><span class=pl-c1>constructor</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>monitorXHR</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>req</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>_open</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_open</span> <span class=pl-c1>=</span> <span class=pl-s1>req</span><span class=pl-kos>.</span><span class=pl-c1>open</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>req</span><span class=pl-kos>.</span><span class=pl-en>open</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>type</span><span class=pl-kos>,</span> <span class=pl-s1>url</span><span class=pl-kos>,</span> <span class=pl-s1>async</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>shouldTrack</span><span class=pl-kos>(</span><span class=pl-s1>type</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;request&#39;</span><span class=pl-kos>,</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>type</span>: <span class=pl-s1>type</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>url</span>: <span class=pl-s1>url</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>request</span>: <span class=pl-s1>req</span></td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>_open</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-s1>req</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-en>XMLHttpRequest</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>flags</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>req</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>_XMLHttpRequest</span><span class=pl-kos>(</span><span class=pl-s1>flags</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>monitorXHR</span><span class=pl-kos>(</span><span class=pl-s1>req</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>extendNative</span><span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>XMLHttpRequest</span><span class=pl-kos>,</span> <span class=pl-s1>_XMLHttpRequest</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>_XDomainRequest</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-en>XDomainRequest</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>var</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>req</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>_XDomainRequest</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>monitorXHR</span><span class=pl-kos>(</span><span class=pl-s1>req</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>extendNative</span><span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>XDomainRequest</span><span class=pl-kos>,</span> <span class=pl-s1>_XDomainRequest</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_WebSocket</span> != null<span class=pl-kos>)</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ajax</span><span class=pl-kos>.</span><span class=pl-c1>trackWebSockets</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-en>WebSocket</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>,</span> <span class=pl-s1>protocols</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>var</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>protocols</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>req</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>_WebSocket</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>,</span> <span class=pl-s1>protocols</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>req</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>_WebSocket</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>shouldTrack</span><span class=pl-kos>(</span><span class=pl-s>&#39;socket&#39;</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;request&#39;</span><span class=pl-kos>,</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>type</span>: <span class=pl-s>&#39;socket&#39;</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>url</span>: <span class=pl-s1>url</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>protocols</span>: <span class=pl-s1>protocols</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">              <span class=pl-c1>request</span>: <span class=pl-s1>req</span></td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>req</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>extendNative</span><span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>WebSocket</span><span class=pl-kos>,</span> <span class=pl-s1>_WebSocket</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>RequestIntercept</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-v>Events</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>_intercept</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>getIntercept</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>_intercept</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_intercept</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>RequestIntercept</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>_intercept</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>shouldIgnoreURL</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>pattern</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ajax</span><span class=pl-kos>.</span><span class=pl-c1>ignoreURLs</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>pattern</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>pattern</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;string&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>.</span><span class=pl-en>indexOf</span><span class=pl-kos>(</span><span class=pl-s1>pattern</span><span class=pl-kos>)</span> !== <span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>pattern</span><span class=pl-kos>.</span><span class=pl-en>test</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>getIntercept</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&#39;request&#39;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_arg</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>after</span><span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>,</span> <span class=pl-s1>request</span><span class=pl-kos>,</span> <span class=pl-s1>type</span><span class=pl-kos>,</span> <span class=pl-s1>url</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>type</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>type</span><span class=pl-kos>,</span> <span class=pl-s1>request</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>request</span><span class=pl-kos>,</span> <span class=pl-s1>url</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>url</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>shouldIgnoreURL</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>restartOnRequestAfter</span> !== <span class=pl-c1>false</span> <span class=pl-c1>||</span> <span class=pl-s1>shouldTrack</span><span class=pl-kos>(</span><span class=pl-s1>type</span><span class=pl-kos>)</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;force&#39;</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>args</span> <span class=pl-c1>=</span> <span class=pl-smi>arguments</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>after</span> <span class=pl-c1>=</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>restartOnRequestAfter</span> <span class=pl-c1>||</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>after</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;boolean&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>after</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>stillActive</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span> <span class=pl-s1>_ref3</span><span class=pl-kos>,</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>type</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;socket&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>stillActive</span> <span class=pl-c1>=</span> <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>stillActive</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-c1>0</span> <span class=pl-c1>&lt;</span> <span class=pl-kos>(</span><span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span><span class=pl-kos>)</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>_ref2</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>4</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>stillActive</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">          <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>restart</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_ref3</span> <span class=pl-c1>=</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>sources</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_results</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref3</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref3</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>source</span> <span class=pl-k>instanceof</span> <span class=pl-v>AjaxMonitor</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line">              <span class=pl-s1>source</span><span class=pl-kos>.</span><span class=pl-c1>watch</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-s1>source</span><span class=pl-kos>,</span> <span class=pl-s1>args</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line">              <span class=pl-k>break</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line">              <span class=pl-s1>_results</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>_results</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-s1>after</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>AjaxMonitor</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>AjaxMonitor</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>elements</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>getIntercept</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&#39;request&#39;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>watch</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-s1>_this</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>AjaxMonitor</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>watch</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_arg</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>request</span><span class=pl-kos>,</span> <span class=pl-s1>tracker</span><span class=pl-kos>,</span> <span class=pl-s1>type</span><span class=pl-kos>,</span> <span class=pl-s1>url</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>type</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>type</span><span class=pl-kos>,</span> <span class=pl-s1>request</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>request</span><span class=pl-kos>,</span> <span class=pl-s1>url</span> <span class=pl-c1>=</span> <span class=pl-s1>_arg</span><span class=pl-kos>.</span><span class=pl-c1>url</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>shouldIgnoreURL</span><span class=pl-kos>(</span><span class=pl-s1>url</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>type</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;socket&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>tracker</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>SocketRequestTracker</span><span class=pl-kos>(</span><span class=pl-s1>request</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>tracker</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>XHRRequestTracker</span><span class=pl-kos>(</span><span class=pl-s1>request</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>elements</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>tracker</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>AjaxMonitor</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>XHRRequestTracker</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>XHRRequestTracker</span><span class=pl-kos>(</span><span class=pl-s1>request</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>size</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_onreadystatechange</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>ProgressEvent</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>size</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-en>addEventListener</span><span class=pl-kos>(</span><span class=pl-s>&#39;progress&#39;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>evt</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>evt</span><span class=pl-kos>.</span><span class=pl-c1>lengthComputable</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span> * <span class=pl-s1>evt</span><span class=pl-kos>.</span><span class=pl-c1>loaded</span> / <span class=pl-s1>evt</span><span class=pl-kos>.</span><span class=pl-c1>total</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-c1>100</span> <span class=pl-c1>-</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span> / <span class=pl-c1>2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>false</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s>&#39;load&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;abort&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;timeout&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;error&#39;</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>event</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-en>addEventListener</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>false</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_onreadystatechange</span> <span class=pl-c1>=</span> <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-c1>onreadystatechange</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-en>onreadystatechange</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>var</span> <span class=pl-s1>_ref3</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>_ref3</span> <span class=pl-c1>=</span> <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span><span class=pl-kos>)</span> <span class=pl-c1>===</span> <span class=pl-c1>0</span> <span class=pl-c1>||</span> <span class=pl-s1>_ref3</span> <span class=pl-c1>===</span> <span class=pl-c1>4</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span> <span class=pl-c1>===</span> <span class=pl-c1>3</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>50</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-k>typeof</span> <span class=pl-s1>_onreadystatechange</span> <span class=pl-c1>===</span> <span class=pl-s>&quot;function&quot;</span> ? <span class=pl-s1>_onreadystatechange</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>XHRRequestTracker</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>SocketRequestTracker</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>SocketRequestTracker</span><span class=pl-kos>(</span><span class=pl-s1>request</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s>&#39;error&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;open&#39;</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>event</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>request</span><span class=pl-kos>.</span><span class=pl-en>addEventListener</span><span class=pl-kos>(</span><span class=pl-s1>event</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>false</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>SocketRequestTracker</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>ElementMonitor</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>ElementMonitor</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>selector</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>options</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>elements</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>selectors</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>selectors</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>selectors</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>selector</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>elements</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-k>new</span> <span class=pl-v>ElementTracker</span><span class=pl-kos>(</span><span class=pl-s1>selector</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>ElementMonitor</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>ElementTracker</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>ElementTracker</span><span class=pl-kos>(</span><span class=pl-s1>selector</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>selector</span> <span class=pl-c1>=</span> <span class=pl-s1>selector</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>check</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>ElementTracker</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>check</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>querySelector</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>selector</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>done</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-en>check</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>elements</span><span class=pl-kos>.</span><span class=pl-c1>checkInterval</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>ElementTracker</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>done</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>ElementTracker</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L678" class="blob-num js-line-number" data-line-number="678"></td>
        <td id="LC678" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L679" class="blob-num js-line-number" data-line-number="679"></td>
        <td id="LC679" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L680" class="blob-num js-line-number" data-line-number="680"></td>
        <td id="LC680" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L681" class="blob-num js-line-number" data-line-number="681"></td>
        <td id="LC681" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>DocumentMonitor</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L682" class="blob-num js-line-number" data-line-number="682"></td>
        <td id="LC682" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>DocumentMonitor</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-c1>states</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L683" class="blob-num js-line-number" data-line-number="683"></td>
        <td id="LC683" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>loading</span>: <span class=pl-c1>0</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L684" class="blob-num js-line-number" data-line-number="684"></td>
        <td id="LC684" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>interactive</span>: <span class=pl-c1>50</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L685" class="blob-num js-line-number" data-line-number="685"></td>
        <td id="LC685" class="blob-code blob-code-inner js-file-line">      <span class=pl-c1>complete</span>: <span class=pl-c1>100</span></td>
      </tr>
      <tr>
        <td id="L686" class="blob-num js-line-number" data-line-number="686"></td>
        <td id="LC686" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L687" class="blob-num js-line-number" data-line-number="687"></td>
        <td id="LC687" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L688" class="blob-num js-line-number" data-line-number="688"></td>
        <td id="LC688" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>DocumentMonitor</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L689" class="blob-num js-line-number" data-line-number="689"></td>
        <td id="LC689" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>_onreadystatechange</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L690" class="blob-num js-line-number" data-line-number="690"></td>
        <td id="LC690" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L691" class="blob-num js-line-number" data-line-number="691"></td>
        <td id="LC691" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>states</span><span class=pl-kos>[</span><span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span><span class=pl-kos>]</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref2</span> : <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L692" class="blob-num js-line-number" data-line-number="692"></td>
        <td id="LC692" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>_onreadystatechange</span> <span class=pl-c1>=</span> <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>onreadystatechange</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L693" class="blob-num js-line-number" data-line-number="693"></td>
        <td id="LC693" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>onreadystatechange</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L694" class="blob-num js-line-number" data-line-number="694"></td>
        <td id="LC694" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>states</span><span class=pl-kos>[</span><span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span><span class=pl-kos>]</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L695" class="blob-num js-line-number" data-line-number="695"></td>
        <td id="LC695" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>states</span><span class=pl-kos>[</span><span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-c1>readyState</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L696" class="blob-num js-line-number" data-line-number="696"></td>
        <td id="LC696" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L697" class="blob-num js-line-number" data-line-number="697"></td>
        <td id="LC697" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-k>typeof</span> <span class=pl-s1>_onreadystatechange</span> <span class=pl-c1>===</span> <span class=pl-s>&quot;function&quot;</span> ? <span class=pl-s1>_onreadystatechange</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span> : <span class=pl-k>void</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L698" class="blob-num js-line-number" data-line-number="698"></td>
        <td id="LC698" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L699" class="blob-num js-line-number" data-line-number="699"></td>
        <td id="LC699" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L700" class="blob-num js-line-number" data-line-number="700"></td>
        <td id="LC700" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L701" class="blob-num js-line-number" data-line-number="701"></td>
        <td id="LC701" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>DocumentMonitor</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L702" class="blob-num js-line-number" data-line-number="702"></td>
        <td id="LC702" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L703" class="blob-num js-line-number" data-line-number="703"></td>
        <td id="LC703" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L704" class="blob-num js-line-number" data-line-number="704"></td>
        <td id="LC704" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L705" class="blob-num js-line-number" data-line-number="705"></td>
        <td id="LC705" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>EventLagMonitor</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L706" class="blob-num js-line-number" data-line-number="706"></td>
        <td id="LC706" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>EventLagMonitor</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L707" class="blob-num js-line-number" data-line-number="707"></td>
        <td id="LC707" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>avg</span><span class=pl-kos>,</span> <span class=pl-s1>interval</span><span class=pl-kos>,</span> <span class=pl-s1>last</span><span class=pl-kos>,</span> <span class=pl-s1>points</span><span class=pl-kos>,</span> <span class=pl-s1>samples</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L708" class="blob-num js-line-number" data-line-number="708"></td>
        <td id="LC708" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_this</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L709" class="blob-num js-line-number" data-line-number="709"></td>
        <td id="LC709" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L710" class="blob-num js-line-number" data-line-number="710"></td>
        <td id="LC710" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>avg</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L711" class="blob-num js-line-number" data-line-number="711"></td>
        <td id="LC711" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>samples</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L712" class="blob-num js-line-number" data-line-number="712"></td>
        <td id="LC712" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>points</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L713" class="blob-num js-line-number" data-line-number="713"></td>
        <td id="LC713" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>last</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L714" class="blob-num js-line-number" data-line-number="714"></td>
        <td id="LC714" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>interval</span> <span class=pl-c1>=</span> <span class=pl-en>setInterval</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L715" class="blob-num js-line-number" data-line-number="715"></td>
        <td id="LC715" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>diff</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L716" class="blob-num js-line-number" data-line-number="716"></td>
        <td id="LC716" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>diff</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>last</span> <span class=pl-c1>-</span> <span class=pl-c1>50</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L717" class="blob-num js-line-number" data-line-number="717"></td>
        <td id="LC717" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>last</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L718" class="blob-num js-line-number" data-line-number="718"></td>
        <td id="LC718" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>samples</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>diff</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L719" class="blob-num js-line-number" data-line-number="719"></td>
        <td id="LC719" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>samples</span><span class=pl-kos>.</span><span class=pl-c1>length</span> <span class=pl-c1>&gt;</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>eventLag</span><span class=pl-kos>.</span><span class=pl-c1>sampleCount</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L720" class="blob-num js-line-number" data-line-number="720"></td>
        <td id="LC720" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>samples</span><span class=pl-kos>.</span><span class=pl-en>shift</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L721" class="blob-num js-line-number" data-line-number="721"></td>
        <td id="LC721" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L722" class="blob-num js-line-number" data-line-number="722"></td>
        <td id="LC722" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>avg</span> <span class=pl-c1>=</span> <span class=pl-s1>avgAmplitude</span><span class=pl-kos>(</span><span class=pl-s1>samples</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L723" class="blob-num js-line-number" data-line-number="723"></td>
        <td id="LC723" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-c1>++</span><span class=pl-s1>points</span> &gt;= <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>eventLag</span><span class=pl-kos>.</span><span class=pl-c1>minSamples</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>avg</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>eventLag</span><span class=pl-kos>.</span><span class=pl-c1>lagThreshold</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L724" class="blob-num js-line-number" data-line-number="724"></td>
        <td id="LC724" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L725" class="blob-num js-line-number" data-line-number="725"></td>
        <td id="LC725" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-en>clearInterval</span><span class=pl-kos>(</span><span class=pl-s1>interval</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L726" class="blob-num js-line-number" data-line-number="726"></td>
        <td id="LC726" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L727" class="blob-num js-line-number" data-line-number="727"></td>
        <td id="LC727" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-s1>_this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span> * <span class=pl-kos>(</span><span class=pl-c1>3</span> / <span class=pl-kos>(</span><span class=pl-s1>avg</span> <span class=pl-c1>+</span> <span class=pl-c1>3</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L728" class="blob-num js-line-number" data-line-number="728"></td>
        <td id="LC728" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L729" class="blob-num js-line-number" data-line-number="729"></td>
        <td id="LC729" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>50</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L730" class="blob-num js-line-number" data-line-number="730"></td>
        <td id="LC730" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L731" class="blob-num js-line-number" data-line-number="731"></td>
        <td id="LC731" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L732" class="blob-num js-line-number" data-line-number="732"></td>
        <td id="LC732" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>EventLagMonitor</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L733" class="blob-num js-line-number" data-line-number="733"></td>
        <td id="LC733" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L734" class="blob-num js-line-number" data-line-number="734"></td>
        <td id="LC734" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L735" class="blob-num js-line-number" data-line-number="735"></td>
        <td id="LC735" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L736" class="blob-num js-line-number" data-line-number="736"></td>
        <td id="LC736" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Scaler</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L737" class="blob-num js-line-number" data-line-number="737"></td>
        <td id="LC737" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-v>Scaler</span><span class=pl-kos>(</span><span class=pl-s1>source</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L738" class="blob-num js-line-number" data-line-number="738"></td>
        <td id="LC738" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>source</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L739" class="blob-num js-line-number" data-line-number="739"></td>
        <td id="LC739" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>last</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>sinceLastUpdate</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L740" class="blob-num js-line-number" data-line-number="740"></td>
        <td id="LC740" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>rate</span> <span class=pl-c1>=</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>initialRate</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L741" class="blob-num js-line-number" data-line-number="741"></td>
        <td id="LC741" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>catchup</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L742" class="blob-num js-line-number" data-line-number="742"></td>
        <td id="LC742" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastProgress</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L743" class="blob-num js-line-number" data-line-number="743"></td>
        <td id="LC743" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>source</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L744" class="blob-num js-line-number" data-line-number="744"></td>
        <td id="LC744" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-s1>result</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>source</span><span class=pl-kos>,</span> <span class=pl-s>&#39;progress&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L745" class="blob-num js-line-number" data-line-number="745"></td>
        <td id="LC745" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L746" class="blob-num js-line-number" data-line-number="746"></td>
        <td id="LC746" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L747" class="blob-num js-line-number" data-line-number="747"></td>
        <td id="LC747" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L748" class="blob-num js-line-number" data-line-number="748"></td>
        <td id="LC748" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Scaler</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>tick</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>frameTime</span><span class=pl-kos>,</span> <span class=pl-s1>val</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L749" class="blob-num js-line-number" data-line-number="749"></td>
        <td id="LC749" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>scaling</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L750" class="blob-num js-line-number" data-line-number="750"></td>
        <td id="LC750" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> <span class=pl-c1>==</span> null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L751" class="blob-num js-line-number" data-line-number="751"></td>
        <td id="LC751" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>val</span> <span class=pl-c1>=</span> <span class=pl-s1>result</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>source</span><span class=pl-kos>,</span> <span class=pl-s>&#39;progress&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L752" class="blob-num js-line-number" data-line-number="752"></td>
        <td id="LC752" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L753" class="blob-num js-line-number" data-line-number="753"></td>
        <td id="LC753" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> &gt;= <span class=pl-c1>100</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L754" class="blob-num js-line-number" data-line-number="754"></td>
        <td id="LC754" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>done</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L755" class="blob-num js-line-number" data-line-number="755"></td>
        <td id="LC755" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L756" class="blob-num js-line-number" data-line-number="756"></td>
        <td id="LC756" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> <span class=pl-c1>===</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>last</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L757" class="blob-num js-line-number" data-line-number="757"></td>
        <td id="LC757" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>sinceLastUpdate</span> <span class=pl-c1>+=</span> <span class=pl-s1>frameTime</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L758" class="blob-num js-line-number" data-line-number="758"></td>
        <td id="LC758" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L759" class="blob-num js-line-number" data-line-number="759"></td>
        <td id="LC759" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>sinceLastUpdate</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L760" class="blob-num js-line-number" data-line-number="760"></td>
        <td id="LC760" class="blob-code blob-code-inner js-file-line">          <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>rate</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> <span class=pl-c1>-</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>last</span><span class=pl-kos>)</span> / <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>sinceLastUpdate</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L761" class="blob-num js-line-number" data-line-number="761"></td>
        <td id="LC761" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L762" class="blob-num js-line-number" data-line-number="762"></td>
        <td id="LC762" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>catchup</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> <span class=pl-c1>-</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span> / <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>catchupTime</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L763" class="blob-num js-line-number" data-line-number="763"></td>
        <td id="LC763" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>sinceLastUpdate</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L764" class="blob-num js-line-number" data-line-number="764"></td>
        <td id="LC764" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>last</span> <span class=pl-c1>=</span> <span class=pl-s1>val</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L765" class="blob-num js-line-number" data-line-number="765"></td>
        <td id="LC765" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L766" class="blob-num js-line-number" data-line-number="766"></td>
        <td id="LC766" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>val</span> <span class=pl-c1>&gt;</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L767" class="blob-num js-line-number" data-line-number="767"></td>
        <td id="LC767" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>+=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>catchup</span> * <span class=pl-s1>frameTime</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L768" class="blob-num js-line-number" data-line-number="768"></td>
        <td id="LC768" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L769" class="blob-num js-line-number" data-line-number="769"></td>
        <td id="LC769" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>scaling</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> / <span class=pl-c1>100</span><span class=pl-kos>,</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>easeFactor</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L770" class="blob-num js-line-number" data-line-number="770"></td>
        <td id="LC770" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>+=</span> <span class=pl-s1>scaling</span> * <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>rate</span> * <span class=pl-s1>frameTime</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L771" class="blob-num js-line-number" data-line-number="771"></td>
        <td id="LC771" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastProgress</span> <span class=pl-c1>+</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>maxProgressPerFrame</span><span class=pl-kos>,</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L772" class="blob-num js-line-number" data-line-number="772"></td>
        <td id="LC772" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>max</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L773" class="blob-num js-line-number" data-line-number="773"></td>
        <td id="LC773" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-c1>100</span><span class=pl-kos>,</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L774" class="blob-num js-line-number" data-line-number="774"></td>
        <td id="LC774" class="blob-code blob-code-inner js-file-line">      <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>lastProgress</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L775" class="blob-num js-line-number" data-line-number="775"></td>
        <td id="LC775" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L776" class="blob-num js-line-number" data-line-number="776"></td>
        <td id="LC776" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L777" class="blob-num js-line-number" data-line-number="777"></td>
        <td id="LC777" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L778" class="blob-num js-line-number" data-line-number="778"></td>
        <td id="LC778" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Scaler</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L779" class="blob-num js-line-number" data-line-number="779"></td>
        <td id="LC779" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L780" class="blob-num js-line-number" data-line-number="780"></td>
        <td id="LC780" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L781" class="blob-num js-line-number" data-line-number="781"></td>
        <td id="LC781" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L782" class="blob-num js-line-number" data-line-number="782"></td>
        <td id="LC782" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>sources</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L783" class="blob-num js-line-number" data-line-number="783"></td>
        <td id="LC783" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L784" class="blob-num js-line-number" data-line-number="784"></td>
        <td id="LC784" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>scalers</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L785" class="blob-num js-line-number" data-line-number="785"></td>
        <td id="LC785" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L786" class="blob-num js-line-number" data-line-number="786"></td>
        <td id="LC786" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>bar</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L787" class="blob-num js-line-number" data-line-number="787"></td>
        <td id="LC787" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L788" class="blob-num js-line-number" data-line-number="788"></td>
        <td id="LC788" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>uniScaler</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L789" class="blob-num js-line-number" data-line-number="789"></td>
        <td id="LC789" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L790" class="blob-num js-line-number" data-line-number="790"></td>
        <td id="LC790" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>animation</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L791" class="blob-num js-line-number" data-line-number="791"></td>
        <td id="LC791" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L792" class="blob-num js-line-number" data-line-number="792"></td>
        <td id="LC792" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>cancelAnimation</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L793" class="blob-num js-line-number" data-line-number="793"></td>
        <td id="LC793" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L794" class="blob-num js-line-number" data-line-number="794"></td>
        <td id="LC794" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L795" class="blob-num js-line-number" data-line-number="795"></td>
        <td id="LC795" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L796" class="blob-num js-line-number" data-line-number="796"></td>
        <td id="LC796" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>handlePushState</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L797" class="blob-num js-line-number" data-line-number="797"></td>
        <td id="LC797" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>restartOnPushState</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L798" class="blob-num js-line-number" data-line-number="798"></td>
        <td id="LC798" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>restart</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L799" class="blob-num js-line-number" data-line-number="799"></td>
        <td id="LC799" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L800" class="blob-num js-line-number" data-line-number="800"></td>
        <td id="LC800" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L801" class="blob-num js-line-number" data-line-number="801"></td>
        <td id="LC801" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L802" class="blob-num js-line-number" data-line-number="802"></td>
        <td id="LC802" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-c1>pushState</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L803" class="blob-num js-line-number" data-line-number="803"></td>
        <td id="LC803" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_pushState</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-c1>pushState</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L804" class="blob-num js-line-number" data-line-number="804"></td>
        <td id="LC804" class="blob-code blob-code-inner js-file-line">    <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-en>pushState</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L805" class="blob-num js-line-number" data-line-number="805"></td>
        <td id="LC805" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>handlePushState</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L806" class="blob-num js-line-number" data-line-number="806"></td>
        <td id="LC806" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>_pushState</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L807" class="blob-num js-line-number" data-line-number="807"></td>
        <td id="LC807" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L808" class="blob-num js-line-number" data-line-number="808"></td>
        <td id="LC808" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L809" class="blob-num js-line-number" data-line-number="809"></td>
        <td id="LC809" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L810" class="blob-num js-line-number" data-line-number="810"></td>
        <td id="LC810" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-c1>replaceState</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L811" class="blob-num js-line-number" data-line-number="811"></td>
        <td id="LC811" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_replaceState</span> <span class=pl-c1>=</span> <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-c1>replaceState</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L812" class="blob-num js-line-number" data-line-number="812"></td>
        <td id="LC812" class="blob-code blob-code-inner js-file-line">    <span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>.</span><span class=pl-en>replaceState</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L813" class="blob-num js-line-number" data-line-number="813"></td>
        <td id="LC813" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>handlePushState</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L814" class="blob-num js-line-number" data-line-number="814"></td>
        <td id="LC814" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>_replaceState</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span><span class=pl-smi>window</span><span class=pl-kos>.</span><span class=pl-c1>history</span><span class=pl-kos>,</span> <span class=pl-smi>arguments</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L815" class="blob-num js-line-number" data-line-number="815"></td>
        <td id="LC815" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L816" class="blob-num js-line-number" data-line-number="816"></td>
        <td id="LC816" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L817" class="blob-num js-line-number" data-line-number="817"></td>
        <td id="LC817" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L818" class="blob-num js-line-number" data-line-number="818"></td>
        <td id="LC818" class="blob-code blob-code-inner js-file-line">  <span class=pl-c1>SOURCE_KEYS</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L819" class="blob-num js-line-number" data-line-number="819"></td>
        <td id="LC819" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>ajax</span>: <span class=pl-v>AjaxMonitor</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L820" class="blob-num js-line-number" data-line-number="820"></td>
        <td id="LC820" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>elements</span>: <span class=pl-v>ElementMonitor</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L821" class="blob-num js-line-number" data-line-number="821"></td>
        <td id="LC821" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>document</span>: <span class=pl-v>DocumentMonitor</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L822" class="blob-num js-line-number" data-line-number="822"></td>
        <td id="LC822" class="blob-code blob-code-inner js-file-line">    <span class=pl-c1>eventLag</span>: <span class=pl-v>EventLagMonitor</span></td>
      </tr>
      <tr>
        <td id="L823" class="blob-num js-line-number" data-line-number="823"></td>
        <td id="LC823" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L824" class="blob-num js-line-number" data-line-number="824"></td>
        <td id="LC824" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L825" class="blob-num js-line-number" data-line-number="825"></td>
        <td id="LC825" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>(</span><span class=pl-s1>init</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L826" class="blob-num js-line-number" data-line-number="826"></td>
        <td id="LC826" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>type</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_k</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_len2</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>,</span> <span class=pl-s1>_ref3</span><span class=pl-kos>,</span> <span class=pl-s1>_ref4</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L827" class="blob-num js-line-number" data-line-number="827"></td>
        <td id="LC827" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>sources</span> <span class=pl-c1>=</span> <span class=pl-s1>sources</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L828" class="blob-num js-line-number" data-line-number="828"></td>
        <td id="LC828" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s>&#39;ajax&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;elements&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;document&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;eventLag&#39;</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L829" class="blob-num js-line-number" data-line-number="829"></td>
        <td id="LC829" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L830" class="blob-num js-line-number" data-line-number="830"></td>
        <td id="LC830" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>type</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref2</span><span class=pl-kos>[</span><span class=pl-s1>_j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L831" class="blob-num js-line-number" data-line-number="831"></td>
        <td id="LC831" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>[</span><span class=pl-s1>type</span><span class=pl-kos>]</span> !== <span class=pl-c1>false</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L832" class="blob-num js-line-number" data-line-number="832"></td>
        <td id="LC832" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>sources</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-k>new</span> <span class=pl-c1>SOURCE_KEYS</span><span class=pl-kos>[</span><span class=pl-s1>type</span><span class=pl-kos>]</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>[</span><span class=pl-s1>type</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L833" class="blob-num js-line-number" data-line-number="833"></td>
        <td id="LC833" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L834" class="blob-num js-line-number" data-line-number="834"></td>
        <td id="LC834" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L835" class="blob-num js-line-number" data-line-number="835"></td>
        <td id="LC835" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_ref4</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>_ref3</span> <span class=pl-c1>=</span> <span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>extraSources</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref3</span> : <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L836" class="blob-num js-line-number" data-line-number="836"></td>
        <td id="LC836" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>_k</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len2</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref4</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_k</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len2</span><span class=pl-kos>;</span> <span class=pl-s1>_k</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L837" class="blob-num js-line-number" data-line-number="837"></td>
        <td id="LC837" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>_ref4</span><span class=pl-kos>[</span><span class=pl-s1>_k</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L838" class="blob-num js-line-number" data-line-number="838"></td>
        <td id="LC838" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>sources</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-k>new</span> <span class=pl-s1>source</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L839" class="blob-num js-line-number" data-line-number="839"></td>
        <td id="LC839" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L840" class="blob-num js-line-number" data-line-number="840"></td>
        <td id="LC840" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>bar</span> <span class=pl-c1>=</span> <span class=pl-s1>bar</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>Bar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L841" class="blob-num js-line-number" data-line-number="841"></td>
        <td id="LC841" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>scalers</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L842" class="blob-num js-line-number" data-line-number="842"></td>
        <td id="LC842" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>uniScaler</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>Scaler</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L843" class="blob-num js-line-number" data-line-number="843"></td>
        <td id="LC843" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L844" class="blob-num js-line-number" data-line-number="844"></td>
        <td id="LC844" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L845" class="blob-num js-line-number" data-line-number="845"></td>
        <td id="LC845" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>stop</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L846" class="blob-num js-line-number" data-line-number="846"></td>
        <td id="LC846" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;stop&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L847" class="blob-num js-line-number" data-line-number="847"></td>
        <td id="LC847" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L848" class="blob-num js-line-number" data-line-number="848"></td>
        <td id="LC848" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>destroy</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L849" class="blob-num js-line-number" data-line-number="849"></td>
        <td id="LC849" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cancelAnimation</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L850" class="blob-num js-line-number" data-line-number="850"></td>
        <td id="LC850" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>animation</span> != null<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L851" class="blob-num js-line-number" data-line-number="851"></td>
        <td id="LC851" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>cancelAnimationFrame</span> <span class=pl-c1>===</span> <span class=pl-s>&quot;function&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L852" class="blob-num js-line-number" data-line-number="852"></td>
        <td id="LC852" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>cancelAnimationFrame</span><span class=pl-kos>(</span><span class=pl-s1>animation</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L853" class="blob-num js-line-number" data-line-number="853"></td>
        <td id="LC853" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L854" class="blob-num js-line-number" data-line-number="854"></td>
        <td id="LC854" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>animation</span> <span class=pl-c1>=</span> null<span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L855" class="blob-num js-line-number" data-line-number="855"></td>
        <td id="LC855" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L856" class="blob-num js-line-number" data-line-number="856"></td>
        <td id="LC856" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>init</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L857" class="blob-num js-line-number" data-line-number="857"></td>
        <td id="LC857" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L858" class="blob-num js-line-number" data-line-number="858"></td>
        <td id="LC858" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L859" class="blob-num js-line-number" data-line-number="859"></td>
        <td id="LC859" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>restart</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L860" class="blob-num js-line-number" data-line-number="860"></td>
        <td id="LC860" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;restart&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L861" class="blob-num js-line-number" data-line-number="861"></td>
        <td id="LC861" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>stop</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L862" class="blob-num js-line-number" data-line-number="862"></td>
        <td id="LC862" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>start</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L863" class="blob-num js-line-number" data-line-number="863"></td>
        <td id="LC863" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L864" class="blob-num js-line-number" data-line-number="864"></td>
        <td id="LC864" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L865" class="blob-num js-line-number" data-line-number="865"></td>
        <td id="LC865" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>go</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L866" class="blob-num js-line-number" data-line-number="866"></td>
        <td id="LC866" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>start</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L867" class="blob-num js-line-number" data-line-number="867"></td>
        <td id="LC867" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L868" class="blob-num js-line-number" data-line-number="868"></td>
        <td id="LC868" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>render</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L869" class="blob-num js-line-number" data-line-number="869"></td>
        <td id="LC869" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>start</span> <span class=pl-c1>=</span> <span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L870" class="blob-num js-line-number" data-line-number="870"></td>
        <td id="LC870" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cancelAnimation</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L871" class="blob-num js-line-number" data-line-number="871"></td>
        <td id="LC871" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>animation</span> <span class=pl-c1>=</span> <span class=pl-s1>runAnimation</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>frameTime</span><span class=pl-kos>,</span> <span class=pl-s1>enqueueNextFrame</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L872" class="blob-num js-line-number" data-line-number="872"></td>
        <td id="LC872" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>avg</span><span class=pl-kos>,</span> <span class=pl-s1>count</span><span class=pl-kos>,</span> <span class=pl-s1>done</span><span class=pl-kos>,</span> <span class=pl-s1>element</span><span class=pl-kos>,</span> <span class=pl-s1>elements</span><span class=pl-kos>,</span> <span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-s1>j</span><span class=pl-kos>,</span> <span class=pl-s1>remaining</span><span class=pl-kos>,</span> <span class=pl-s1>scaler</span><span class=pl-kos>,</span> <span class=pl-s1>scalerList</span><span class=pl-kos>,</span> <span class=pl-s1>sum</span><span class=pl-kos>,</span> <span class=pl-s1>_j</span><span class=pl-kos>,</span> <span class=pl-s1>_k</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span><span class=pl-kos>,</span> <span class=pl-s1>_len2</span><span class=pl-kos>,</span> <span class=pl-s1>_ref2</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L873" class="blob-num js-line-number" data-line-number="873"></td>
        <td id="LC873" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>remaining</span> <span class=pl-c1>=</span> <span class=pl-c1>100</span> <span class=pl-c1>-</span> <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-c1>progress</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L874" class="blob-num js-line-number" data-line-number="874"></td>
        <td id="LC874" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>count</span> <span class=pl-c1>=</span> <span class=pl-s1>sum</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L875" class="blob-num js-line-number" data-line-number="875"></td>
        <td id="LC875" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>done</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L876" class="blob-num js-line-number" data-line-number="876"></td>
        <td id="LC876" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-s1>_j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len1</span> <span class=pl-c1>=</span> <span class=pl-s1>sources</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len1</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>++</span><span class=pl-s1>_j</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L877" class="blob-num js-line-number" data-line-number="877"></td>
        <td id="LC877" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>source</span> <span class=pl-c1>=</span> <span class=pl-s1>sources</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L878" class="blob-num js-line-number" data-line-number="878"></td>
        <td id="LC878" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>scalerList</span> <span class=pl-c1>=</span> <span class=pl-s1>scalers</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> != null ? <span class=pl-s1>scalers</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> : <span class=pl-s1>scalers</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L879" class="blob-num js-line-number" data-line-number="879"></td>
        <td id="LC879" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>elements</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>_ref2</span> <span class=pl-c1>=</span> <span class=pl-s1>source</span><span class=pl-kos>.</span><span class=pl-c1>elements</span><span class=pl-kos>)</span> != null ? <span class=pl-s1>_ref2</span> : <span class=pl-kos>[</span><span class=pl-s1>source</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L880" class="blob-num js-line-number" data-line-number="880"></td>
        <td id="LC880" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-s1>_k</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>_len2</span> <span class=pl-c1>=</span> <span class=pl-s1>elements</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>_k</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>_len2</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>++</span><span class=pl-s1>_k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L881" class="blob-num js-line-number" data-line-number="881"></td>
        <td id="LC881" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>element</span> <span class=pl-c1>=</span> <span class=pl-s1>elements</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L882" class="blob-num js-line-number" data-line-number="882"></td>
        <td id="LC882" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>scaler</span> <span class=pl-c1>=</span> <span class=pl-s1>scalerList</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span> != null ? <span class=pl-s1>scalerList</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span> : <span class=pl-s1>scalerList</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>Scaler</span><span class=pl-kos>(</span><span class=pl-s1>element</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L883" class="blob-num js-line-number" data-line-number="883"></td>
        <td id="LC883" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>done</span> &amp;= <span class=pl-s1>scaler</span><span class=pl-kos>.</span><span class=pl-c1>done</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L884" class="blob-num js-line-number" data-line-number="884"></td>
        <td id="LC884" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>scaler</span><span class=pl-kos>.</span><span class=pl-c1>done</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L885" class="blob-num js-line-number" data-line-number="885"></td>
        <td id="LC885" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>continue</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L886" class="blob-num js-line-number" data-line-number="886"></td>
        <td id="LC886" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L887" class="blob-num js-line-number" data-line-number="887"></td>
        <td id="LC887" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>count</span><span class=pl-c1>++</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L888" class="blob-num js-line-number" data-line-number="888"></td>
        <td id="LC888" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>sum</span> <span class=pl-c1>+=</span> <span class=pl-s1>scaler</span><span class=pl-kos>.</span><span class=pl-en>tick</span><span class=pl-kos>(</span><span class=pl-s1>frameTime</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L889" class="blob-num js-line-number" data-line-number="889"></td>
        <td id="LC889" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L890" class="blob-num js-line-number" data-line-number="890"></td>
        <td id="LC890" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L891" class="blob-num js-line-number" data-line-number="891"></td>
        <td id="LC891" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>avg</span> <span class=pl-c1>=</span> <span class=pl-s1>sum</span> / <span class=pl-s1>count</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L892" class="blob-num js-line-number" data-line-number="892"></td>
        <td id="LC892" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>update</span><span class=pl-kos>(</span><span class=pl-s1>uniScaler</span><span class=pl-kos>.</span><span class=pl-en>tick</span><span class=pl-kos>(</span><span class=pl-s1>frameTime</span><span class=pl-kos>,</span> <span class=pl-s1>avg</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L893" class="blob-num js-line-number" data-line-number="893"></td>
        <td id="LC893" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>done</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-c1>||</span> <span class=pl-s1>done</span> <span class=pl-c1>||</span> <span class=pl-s1>cancelAnimation</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L894" class="blob-num js-line-number" data-line-number="894"></td>
        <td id="LC894" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>update</span><span class=pl-kos>(</span><span class=pl-c1>100</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L895" class="blob-num js-line-number" data-line-number="895"></td>
        <td id="LC895" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;done&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L896" class="blob-num js-line-number" data-line-number="896"></td>
        <td id="LC896" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L897" class="blob-num js-line-number" data-line-number="897"></td>
        <td id="LC897" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>finish</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L898" class="blob-num js-line-number" data-line-number="898"></td>
        <td id="LC898" class="blob-code blob-code-inner js-file-line">          <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L899" class="blob-num js-line-number" data-line-number="899"></td>
        <td id="LC899" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>return</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;hide&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L900" class="blob-num js-line-number" data-line-number="900"></td>
        <td id="LC900" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>max</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>ghostTime</span><span class=pl-kos>,</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>max</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>minTime</span> <span class=pl-c1>-</span> <span class=pl-kos>(</span><span class=pl-s1>now</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>start</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L901" class="blob-num js-line-number" data-line-number="901"></td>
        <td id="LC901" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L902" class="blob-num js-line-number" data-line-number="902"></td>
        <td id="LC902" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>enqueueNextFrame</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L903" class="blob-num js-line-number" data-line-number="903"></td>
        <td id="LC903" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L904" class="blob-num js-line-number" data-line-number="904"></td>
        <td id="LC904" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L905" class="blob-num js-line-number" data-line-number="905"></td>
        <td id="LC905" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L906" class="blob-num js-line-number" data-line-number="906"></td>
        <td id="LC906" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L907" class="blob-num js-line-number" data-line-number="907"></td>
        <td id="LC907" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>start</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_options</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L908" class="blob-num js-line-number" data-line-number="908"></td>
        <td id="LC908" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>extend</span><span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>,</span> <span class=pl-s1>_options</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L909" class="blob-num js-line-number" data-line-number="909"></td>
        <td id="LC909" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>running</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L910" class="blob-num js-line-number" data-line-number="910"></td>
        <td id="LC910" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>try</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L911" class="blob-num js-line-number" data-line-number="911"></td>
        <td id="LC911" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>bar</span><span class=pl-kos>.</span><span class=pl-en>render</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L912" class="blob-num js-line-number" data-line-number="912"></td>
        <td id="LC912" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>catch</span> <span class=pl-kos>(</span><span class=pl-s1>_error</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L913" class="blob-num js-line-number" data-line-number="913"></td>
        <td id="LC913" class="blob-code blob-code-inner js-file-line">      <span class=pl-v>NoTargetError</span> <span class=pl-c1>=</span> <span class=pl-s1>_error</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L914" class="blob-num js-line-number" data-line-number="914"></td>
        <td id="LC914" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L915" class="blob-num js-line-number" data-line-number="915"></td>
        <td id="LC915" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-smi>document</span><span class=pl-kos>.</span><span class=pl-en>querySelector</span><span class=pl-kos>(</span><span class=pl-s>&#39;.pace&#39;</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L916" class="blob-num js-line-number" data-line-number="916"></td>
        <td id="LC916" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-en>setTimeout</span><span class=pl-kos>(</span><span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-c1>start</span><span class=pl-kos>,</span> <span class=pl-c1>50</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L917" class="blob-num js-line-number" data-line-number="917"></td>
        <td id="LC917" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L918" class="blob-num js-line-number" data-line-number="918"></td>
        <td id="LC918" class="blob-code blob-code-inner js-file-line">      <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>trigger</span><span class=pl-kos>(</span><span class=pl-s>&#39;start&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L919" class="blob-num js-line-number" data-line-number="919"></td>
        <td id="LC919" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>go</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L920" class="blob-num js-line-number" data-line-number="920"></td>
        <td id="LC920" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L921" class="blob-num js-line-number" data-line-number="921"></td>
        <td id="LC921" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L922" class="blob-num js-line-number" data-line-number="922"></td>
        <td id="LC922" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L923" class="blob-num js-line-number" data-line-number="923"></td>
        <td id="LC923" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>define</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;function&#39;</span> <span class=pl-c1>&amp;&amp;</span> <span class=pl-s1>define</span><span class=pl-kos>.</span><span class=pl-c1>amd</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L924" class="blob-num js-line-number" data-line-number="924"></td>
        <td id="LC924" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>define</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s>&#39;pace&#39;</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L925" class="blob-num js-line-number" data-line-number="925"></td>
        <td id="LC925" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-v>Pace</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L926" class="blob-num js-line-number" data-line-number="926"></td>
        <td id="LC926" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L927" class="blob-num js-line-number" data-line-number="927"></td>
        <td id="LC927" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-s1>exports</span> <span class=pl-c1>===</span> <span class=pl-s>&#39;object&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L928" class="blob-num js-line-number" data-line-number="928"></td>
        <td id="LC928" class="blob-code blob-code-inner js-file-line">    <span class=pl-smi>module</span><span class=pl-kos>.</span><span class=pl-c1>exports</span> <span class=pl-c1>=</span> <span class=pl-v>Pace</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L929" class="blob-num js-line-number" data-line-number="929"></td>
        <td id="LC929" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L930" class="blob-num js-line-number" data-line-number="930"></td>
        <td id="LC930" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>options</span><span class=pl-kos>.</span><span class=pl-c1>startOnPageLoad</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L931" class="blob-num js-line-number" data-line-number="931"></td>
        <td id="LC931" class="blob-code blob-code-inner js-file-line">      <span class=pl-v>Pace</span><span class=pl-kos>.</span><span class=pl-en>start</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L932" class="blob-num js-line-number" data-line-number="932"></td>
        <td id="LC932" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L933" class="blob-num js-line-number" data-line-number="933"></td>
        <td id="LC933" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L934" class="blob-num js-line-number" data-line-number="934"></td>
        <td id="LC934" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L935" class="blob-num js-line-number" data-line-number="935"></td>
        <td id="LC935" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2" style="width:185px">
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;">
            Copy lines
          </clipboard-copy>
        </li>
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;">
            Copy permalink
          </clipboard-copy>
        </li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/HubSpot/pace/blame/4afddfb37aec45cc8588a91416fe28ca173314a8/pace.js">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" role="menuitem" href="/HubSpot/pace/issues/new">Reference in new issue</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  

  <details class="details-reset details-overlay details-overlay-dark">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get">
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>



  </div>
</div>

    </main>
  </div>
  

  </div>

        
<div class="footer container-lg width-full p-responsive" role="contentinfo">
  <div class="position-relative d-flex flex-row-reverse flex-lg-row flex-wrap flex-lg-nowrap flex-justify-center flex-lg-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
      <li class="mr-3 mr-lg-0">&copy; 2020 GitHub, Inc.</li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to security, text:security" href="https://github.com/security">Security</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>

    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon d-none d-lg-block mx-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>
    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" async="async" integrity="sha512-WcQmT2vhcClFVOaaAJV/M+HqsJ2Gq/myvl6F3gCVBxykazXTs+i5fvxncSXwyG1CSfcrqmLFw/R/bmFYzprX2A==" type="application/javascript" id="js-conditional-compat" data-src="https://github.githubassets.com/assets/compat-bootstrap-59c4264f.js"></script>
    <script crossorigin="anonymous" integrity="sha512-Y86V8OBlvF6I/7e56GKOOt80Yg1RTGA09uqFFX18aiBtevLbKGxB7sVpCn79fukppFIBqyBTB/s6l0Bhn0kidQ==" type="application/javascript" src="https://github.githubassets.com/assets/environment-bootstrap-63ce95f0.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-jfR+4VdZuPf5Ck+JA3AZuzWGHz9Sb21keZOYuMoNdfMJovIUb9vxfSdvNSchxAwj5oav48KBfa54+wbuuW8Tlg==" type="application/javascript" src="https://github.githubassets.com/assets/vendor-8df47ee1.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-YAej83ikXSzOu18Y/10VPeKXMyVIJIFTc8h4ZJO+q0GQYefYQUO3WnyTeevbpNDijefJvDgi87QkGOwKLGJ2YA==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-6007a3f3.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-3LAxF4uJ1AzoQQX8Yt1Xo0DpwV+WmyyU2fQ1UdGYNXpDPk7sb7rFfG3XKRgnmuCmys7pyKg+sYib5ZxgfbbOzw==" type="application/javascript" src="https://github.githubassets.com/assets/github-bootstrap-dcb03117.js"></script>
    
      <script crossorigin="anonymous" async="async" integrity="sha512-4GcSWGoe36+BoWho4gtJcByZe8j43w+lt2/PDe3rmBxRVSgD29YipDwuIywe8fvOd2b2CszBqaPGxSznUtE3Xg==" type="application/javascript" data-module-id="./drag-drop.js" data-src="https://github.githubassets.com/assets/drag-drop-e0671258.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-3Vk1NFIOm+TBUMM6pTA6DCUwwLLnc/QIT8jpENm71InvSU8O4p2plDagpst1tH1l+9jOBnneaXZnAskA9a2b3w==" type="application/javascript" data-module-id="./gist-vendor.js" data-src="https://github.githubassets.com/assets/gist-vendor-dd593534.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-urN6bhHnHu4C12A+cTH3dOp+CwLaycy2HUXr95hvu5pbYRdF8z6iR+UQcTZutQ6mZG3Njluw2MTZVCNmwcqh8g==" type="application/javascript" data-module-id="./randomColor.js" data-src="https://github.githubassets.com/assets/randomColor-bab37a6e.js"></script>
    
    
  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>


  </body>
</html>

