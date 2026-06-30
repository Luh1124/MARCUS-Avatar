from __future__ import annotations

import re
from collections.abc import Iterable

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class FaceStudioTheme(Base):
    """
    A dark-themed Gradio theme for FaceStudio applications.

    This theme provides a cohesive dark UI with:
    - Dark neutral backgrounds
    - Muted olive/lime accent colors for selections
    - Modern rounded corners
    - Inter and JetBrains Mono fonts
    """

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.violet,
        secondary_hue: colors.Color | str = colors.purple,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "FaceTex Studio"
        super().set(
            # Colors – use dark-ish neutrals even for the "light" tokens
            background_fill_primary="*neutral_950",  # was *neutral_50
            layout_gap="16px",
            slider_color="*primary_500",
            slider_color_dark="*primary_500",
            # Shadows
            shadow_drop="0 1px 4px 0 rgb(0 0 0 / 0.4)",
            shadow_drop_lg="0 2px 8px 0 rgb(0 0 0 / 0.45)",
            # Block Labels / Blocks – make blocks dark instead of white
            block_background_fill="*neutral_900",  # was white
            block_label_padding="*spacing_sm *spacing_md",
            block_label_background_fill="*primary_600",
            block_label_background_fill_dark="*primary_600",
            block_label_radius="8px",
            block_label_text_size="*text_md",
            block_label_text_weight="600",
            block_label_text_color="white",  # use dark text color
            block_label_text_color_dark="white",
            block_radius="10px",
            block_title_radius="*block_label_radius",
            block_title_padding="*block_label_padding",
            block_title_background_fill="*block_label_background_fill",
            block_title_text_weight="600",
            block_title_text_color="white",  # match dark
            block_title_text_color_dark="white",
            block_label_margin="*spacing_md",
            # Inputs – dark backgrounds, lighter borders
            input_background_fill="*neutral_900",  # was white
            input_background_fill_dark="*neutral_900",  # was white
            input_border_color="*neutral_700",  # was *neutral_50
            input_shadow="*shadow_drop",
            input_shadow_focus="*shadow_drop_lg",
            input_radius="10px",
            checkbox_shadow="none",
            checkbox_border_radius="10px",
            # Buttons
            shadow_spread="6px",
            button_primary_shadow="*shadow_drop_lg",
            button_primary_shadow_hover="*shadow_drop_lg",
            button_primary_shadow_active="*shadow_inset",
            button_secondary_shadow="*shadow_drop_lg",
            button_secondary_shadow_hover="*shadow_drop_lg",
            button_secondary_shadow_active="*shadow_inset",
            checkbox_label_shadow="*shadow_drop_lg",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_400",
            button_primary_background_fill_hover_dark="*primary_500",
            button_primary_text_color="white",
            # Secondary buttons now dark too
            button_secondary_background_fill="*neutral_800",  # was white
            button_secondary_background_fill_hover="*neutral_700",  # was *neutral_100
            button_secondary_background_fill_hover_dark="*neutral_700",
            button_secondary_text_color="white",  # was *neutral_800
            button_cancel_background_fill="*button_secondary_background_fill",
            button_cancel_background_fill_hover="*button_secondary_background_fill_hover",
            button_cancel_background_fill_hover_dark="*button_secondary_background_fill_hover",
            button_cancel_text_color="*button_secondary_text_color",
            # Checkboxes – align light tokens with dark styling
            checkbox_label_background_fill_selected="#2c174a",
            checkbox_label_background_fill_selected_dark="#2c174a",
            checkbox_label_background_fill_dark="*input_background_fill",
            checkbox_label_background_fill_hover_dark="*input_background_fill",
            checkbox_label_text_color_selected_dark="#d6c6ff",
            checkbox_label_text_color_dark="#a3a8b3",
            checkbox_border_width="1px",
            checkbox_border_color="*neutral_600",  # was *neutral_100
            checkbox_border_color_dark="*neutral_600",
            checkbox_background_color_selected="*primary_700",  # match dark-ish
            checkbox_background_color_selected_dark="*primary_700",
            checkbox_border_color_focus="*primary_600",  # match dark
            checkbox_border_color_focus_dark="*primary_600",
            checkbox_border_color_selected="*primary_700",  # match dark
            checkbox_border_color_selected_dark="*primary_700",
            # Borders
            block_border_width="0px",
            panel_border_width="1px",
        )

    def _get_theme_css(self):
        """Generate CSS with dark mode as the default."""
        css = {}
        dark_css = {}

        for attr, val in self.__dict__.items():
            if attr.startswith("_"):
                continue
            if val is None:
                if attr.endswith("_dark"):
                    dark_css[attr[:-5]] = None
                    continue
                else:
                    raise ValueError(
                        f"Cannot set '{attr}' to None - only dark mode variables can be None."
                    )
            val = str(val)
            pattern = r"(\*)([\w_]+)(\b)"

            def repl_func(match):
                full_match = match.group(0)
                if full_match.startswith("*") and full_match.endswith("_dark"):
                    raise ValueError(
                        f"Cannot refer '{attr}' to '{val}' - dark variable references are automatically used for dark mode attributes, so do not use the _dark suffix in the value."
                    )
                if (
                    attr.endswith("_dark")
                    and full_match.startswith("*")
                    and attr[:-5] == full_match[1:]
                ):
                    raise ValueError(
                        f"Cannot refer '{attr}' to '{val}' - if dark and light mode values are the same, set dark mode version to None."
                    )

                word = match.group(2)
                word = word.replace("_", "-")
                return f"var(--{word})"

            val = re.sub(pattern, repl_func, val)

            attr = attr.replace("_", "-")

            if attr.endswith("-dark"):
                attr = attr[:-5]
                dark_css[attr] = val
            else:
                css[attr] = val

        for attr, val in css.items():
            if attr not in dark_css:
                dark_css[attr] = val

        dark_css_code = (
            "\n:root {\n"
            + "\n".join([f"  --{attr}: {val};" for attr, val in dark_css.items()])
            + "\n}"
        )

        font_css = "\n".join(self._font_css)

        return f"{font_css}\n{dark_css_code}"


# Pre-instantiated theme for convenience
face_studio_theme = FaceStudioTheme()
