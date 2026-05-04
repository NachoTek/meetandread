# Override the broken contrib hook for webrtcvad.
# webrtcvad-wheels doesn't have standard metadata, so copy_metadata fails.
# webrtcvad is a single .py file with no metadata — nothing to collect.
