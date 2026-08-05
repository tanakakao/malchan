import React, { useEffect, useState } from "react";

const ICON_FILENAMES = ["icon.png", "icon.svg", "icon.webp", "icon.jpg", "icon.jpeg"];
const ICON_DIRECTORY = `${import.meta.env.BASE_URL}conversation-mode/`;

let resolvedIconUrl;
let iconResolutionPromise;

function probeImage(url) {
  return new Promise((resolve) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(url);
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

async function findConversationIcon() {
  for (const filename of ICON_FILENAMES) {
    const url = `${ICON_DIRECTORY}${filename}`;
    const loadedUrl = await probeImage(url);
    if (loadedUrl) return loadedUrl;
  }
  return null;
}

function resolveConversationIcon() {
  if (resolvedIconUrl !== undefined) {
    return Promise.resolve(resolvedIconUrl);
  }
  if (!iconResolutionPromise) {
    iconResolutionPromise = findConversationIcon().then((url) => {
      resolvedIconUrl = url;
      return url;
    });
  }
  return iconResolutionPromise;
}

export default function ConversationIcon({ fallback = "m", className = "", useImage = true }) {
  const imageEnabled = useImage && fallback !== "自";
  const [iconUrl, setIconUrl] = useState(() => resolvedIconUrl ?? null);

  useEffect(() => {
    let active = true;
    if (!imageEnabled) {
      setIconUrl(null);
      return () => {
        active = false;
      };
    }

    resolveConversationIcon().then((url) => {
      if (active) setIconUrl(url);
    });

    return () => {
      active = false;
    };
  }, [imageEnabled]);

  if (!imageEnabled || !iconUrl) {
    return <span className={className} aria-hidden="true">{fallback}</span>;
  }

  return (
    <span className={`${className} conversation-icon-frame`} aria-hidden="true">
      <img
        className="conversation-icon-image loaded"
        src={iconUrl}
        alt=""
        decoding="async"
      />
    </span>
  );
}
