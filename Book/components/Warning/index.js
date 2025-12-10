import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const Warning = ({children, title = 'Warning'}) => {
  return (
    <div className={clsx(styles.warning, styles['warning--alert'])}>
      <div className={styles.warningTitle}>
        <span className={styles.warningIcon}>⚠️</span>
        {title}
      </div>
      <div className={styles.warningContent}>{children}</div>
    </div>
  );
};

export default Warning;