import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const Callout = ({children, type = 'default', title}) => {
  return (
    <div className={clsx(styles.callout, styles[`callout--${type}`])}>
      {title && <div className={styles.calloutTitle}>{title}</div>}
      <div className={styles.calloutContent}>{children}</div>
    </div>
  );
};

export default Callout;