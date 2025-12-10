import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {isSamePath, useLocalPathname} from '@docusaurus/theme-common';
import {translate} from '@docusaurus/Translate';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDocSidebarItemsExpandedState} from '@docusaurus/plugin-content-docs/client';
import isInternalUrl from '@docusaurus/isInternalUrl';
import styles from './styles.module.css';
import DocSidebarItemCategory from '@theme/DocSidebarItem/Category';
import DocSidebarItemLink from '@theme/DocSidebarItem/Link';
import DocSidebarItemHtml from '@theme/DocSidebarItem/Html';

const DocSidebarItems = ({items, ...props}) => {
  const localPathname = useLocalPathname();
  return (
    <ul className={clsx(ThemeClassNames.docs.docSidebarMenu, styles.menu)}>
      {items.map((item, index) => {
        switch (item.type) {
          case 'category':
            return (
              <li key={index} className={styles.menuItem}>
                <DocSidebarItemCategory item={item} {...props} />
              </li>
            );
          case 'html':
            return (
              <li key={index} className={styles.menuItem} dangerouslySetInnerHTML={{__html: item.value}} />
            );
          case 'link':
          default:
            return (
              <li key={index} className={styles.menuItem}>
                <DocSidebarItemLink
                  item={{
                    ...item,
                    active: isSamePath(item.href, localPathname),
                  }}
                  {...props}
                />
              </li>
            );
        }
      })}
    </ul>
  );
};

export default DocSidebarItems;