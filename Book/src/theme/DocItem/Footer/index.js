import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

function PaginationArrow({url, label, direction}) {
  if (!url) {
    return null;
  }

  const isNext = direction === 'next';

  return (
    <Link
      to={url}
      className={clsx(
        'pagination-nav__link',
        isNext ? 'pagination-nav__link--next' : 'pagination-nav__link--prev',
        styles.paginationArrow,
        styles[`paginationArrow--${direction}`],
      )}>
      <div className="pagination-nav__sublabel">
        {isNext
          ? translate({
              id: 'theme.docs.footer.next',
              message: 'Next',
            })
          : translate({
              id: 'theme.docs.footer.previous',
              message: 'Previous',
            })}
      </div>
      <div className="pagination-nav__label">{label}</div>
    </Link>
  );
}

export default function DocItemFooter() {
  const {metadata} = useDoc();
  const {previous, next} = metadata;

  if (!previous && !next) {
    return null;
  }

  return (
    <nav
      className="pagination-nav docusaurus-mt-lg"
      aria-label={translate({
        id: 'theme.docs.footer.navAriaLabel',
        message: 'Docs page navigation',
      })}>
      {previous && (
        <PaginationArrow
          url={previous.permalink}
          label={previous.title}
          direction="prev"
        />
      )}
      {next && (
        <PaginationArrow
          url={next.permalink}
          label={next.title}
          direction="next"
        />
      )}
    </nav>
  );
}