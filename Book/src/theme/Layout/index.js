import React from 'react';
import DefaultLayout from '@theme-original/Layout';
import RagChatbot from '@site/src/components/RagChatbot';

export default function Layout(props) {
  return (
    <>
      <DefaultLayout {...props}>
        {props.children}
      </DefaultLayout>
      <RagChatbot backendUrl="http://localhost:8001" />
    </>
  );
}