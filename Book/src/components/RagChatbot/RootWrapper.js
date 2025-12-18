import React from 'react';
import RagChatbot from './RagChatbot';

const RootWrapper = ({ children }) => {
  return (
    <>
      {children}
      <RagChatbot backendUrl="http://localhost:8000" />
    </>
  );
};

export default RootWrapper;