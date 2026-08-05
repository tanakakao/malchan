import React, { useEffect, useState } from "react";

const ICON_FILENAMES = ["icon.png", "icon.svg", "icon.webp", "icon.jpg", "icon.jpeg"];
const ICON_DIRECTORY = `${import.meta.env.BASE_URL}conversation-mode/`;

export default function ConversationIcon({ fallback = "m", className = "", useImage = true }) {
  const [index, setIndex] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const imageEnabled = useImage && fallback !== "自";
  const filename = imageEnabled ? ICON_FILENAMES[index] : null;

  useEffect(() => {
    setIndex(0);
    setLoaded(false);
  }, [imageEnabled]);

  useEffect(() => {
    setLoaded(false);
  }, [index]);

  if (!filename) {
    return <span className={className} aria-hidden="true">{fallback}</span>;
  }

  return (
    <span className={`${className} conversation-icon-frame`} aria-hidden="true">
      {!loaded && <span className="conversation-icon-fallback">{fallback}</span>}
      <img
        key={filename}
        className={`conversation-icon-image ${loaded ? "loaded" : ""}`}
        src={`${ICON_DIRECTORY}${filename}`}
        alt=""
        decoding="async"
        onLoad={() => setLoaded(true)}
        onError={() => setIndex((current) => current + 1)}
      />
    </span>
  );
}
