import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const Tip = ({children, title = 'Tip'}) => {
  return (
    <div className={clsx(styles.tip, styles['tip--helpful'])}>
      <div className={styles.tipTitle}>
        <span className={styles.tipIcon}>💡</span>
        {title}
      </div>
      <div className={styles.tipContent}>{children}</div>
    </div>
  );
};

export default Tip;