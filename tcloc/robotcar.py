"""
Oxford RobotCar: turning raw stereo frames into the tcloc data format.

Only the front stereo camera (Bumblebee XB3, wide pair: left and right, 24 cm
apart) is used. Each raw frame is a Bayer image (1280 x 960). Per keyframe:

1. debayer and undistort both images with the RobotCar SDK's lookup tables;
2. keep the top 768 rows (the bonnet is below) and match the pair at full
   resolution (tcloc.stereo: semi-global block matching, keeping consistent,
   textured, non-sky pixels);
3. scale the left image and the disparity to 640 x 384 and convert disparity to
   depth (``fx * baseline / disparity``).

The settings below are those used for the released data
(``metadata/preprocess.json``). Downloads go through the RobotCar data share,
which needs a free account (https://robotcar-dataset.robots.ox.ac.uk), passed as
ROBOTCAR_USERNAME / ROBOTCAR_PASSWORD.
"""

from __future__ import annotations

import re
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from tcloc.stereo import StereoConfig, depth, disparity

RAW_SIZE = (960, 1280)  # rows, cols of a raw stereo image

SDK_URL = (
    "https://raw.githubusercontent.com/ori-mrg/robotcar-dataset-sdk/master"
    "/models/{name}"
)

SHARE = "https://mrgdatashare.robots.ox.ac.uk"


# rows kept from the top of the undistorted image (the bonnet is below)
CROP_ROWS = 768
SCALE = 0.5  # then scaled to 640 x 384


class Camera:
    """
    One stereo camera from the RobotCar SDK.

    Holds its intrinsics, position and undistortion lookup table, read from the
    SDK's ``<name>.txt`` and ``<name>_distortion_lut.bin`` files.

    Args:
        sdk: Folder with the SDK camera files (see :func:`fetch_sdk`).
        name: Camera name, e.g. ``stereo_wide_left``.

    Attributes:
        fx: Focal length along x, in pixels of the raw image.
        fy: Focal length along y, in pixels of the raw image.
        cx: Principal point x, in pixels of the raw image (integer pixel
            centres, as in the SDK).
        cy: Principal point y, in pixels of the raw image (integer pixel
            centres, as in the SDK).
        G_camera_image: (4, 4) float64 transform from the SDK model file;
            its translation (metres) is the camera's position, used for the
            stereo baseline.
        map_x: (960, 1280) float32 undistortion map: for each undistorted
            pixel, the x coordinate to sample in the raw image.
        map_y: (960, 1280) float32 undistortion map: for each undistorted
            pixel, the y coordinate to sample in the raw image.
    """

    def __init__(self, sdk: Path, name: str):
        lines = (sdk / f"{name}.txt").read_text().split("\n")
        self.fx, self.fy, self.cx, self.cy = map(float, lines[0].split())

        self.G_camera_image = np.array(
            [[float(x) for x in line.split()] for line in lines[1:5]]
        )

        lut = (
            np.fromfile(sdk / f"{name}_distortion_lut.bin", np.double)
            .reshape(2, *RAW_SIZE)
            .astype(np.float32)
        )

        self.map_x, self.map_y = lut

    def K(self) -> np.ndarray:
        """
        Intrinsics at output resolution (640 x 384).

        Pixel centres are at +0.5 (the SDK uses integer centres).

        Returns:
            (3, 3) float64 intrinsics K of the cropped, scaled output image.
        """
        return np.array(
            [
                [self.fx * SCALE, 0, (self.cx + 0.5) * SCALE],
                [0, self.fy * SCALE, (self.cy + 0.5) * SCALE],
                [0, 0, 1],
            ]
        )

    def undistort(self, raw: np.ndarray) -> np.ndarray:
        """
        Debayer and undistort a raw image, keeping the top ``CROP_ROWS`` rows.

        Args:
            raw: (960, 1280) uint8 raw Bayer GBRG image.

        Returns:
            (768, 1280, 3) uint8 undistorted RGB image at full resolution
            (pixels mapped from outside the raw image are black).
        """
        rgb = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)

        rgb = cv2.remap(
            rgb,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return rgb[:CROP_ROWS]


def fetch_sdk(directory: Path) -> Path:
    """
    Download the four SDK files of the wide stereo pair.

    The files (CC BY-NC-SA 4.0, Oxford Robotics Institute) are the left and
    right camera models and undistortion lookup tables. Files already present
    are not downloaded again.

    Args:
        directory: Folder to save them in, created if missing.

    Returns:
        ``directory``.

    Raises:
        requests.HTTPError: A download failed.
    """
    directory.mkdir(parents=True, exist_ok=True)

    for side in ("left", "right"):
        for name in (
            f"stereo_wide_{side}.txt",
            f"stereo_wide_{side}_distortion_lut.bin",
        ):
            if not (directory / name).exists():
                r = requests.get(SDK_URL.format(name=name), timeout=120)
                r.raise_for_status()
                (directory / name).write_bytes(r.content)

    return directory


def image_and_depth(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left: Camera,
    right: Camera,
    cfg: StereoConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Output-resolution RGB image and depth of one raw stereo pair.

    Args:
        left_raw: (960, 1280) uint8 raw Bayer GBRG image of the left camera.
        right_raw: (960, 1280) uint8 raw Bayer GBRG image of the right
            camera.
        left: The left camera.
        right: The right camera.
        cfg: Stereo matching settings.

    Returns:
        A tuple ``(image, depth)``: image, (384, 640, 3) uint8 undistorted
        left RGB image; depth, (384, 640) float32 depth of the left image in
        metres, 0 where unknown.
    """
    a, b = left.undistort(left_raw), right.undistort(right_raw)

    d = disparity(
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
        cfg,
    )

    h, w = round(CROP_ROWS * SCALE), round(RAW_SIZE[1] * SCALE)

    baseline = float(
        np.linalg.norm(right.G_camera_image[:3, 3] - left.G_camera_image[:3, 3])
    )

    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA), depth(
        d, left.K()[0, 0], baseline, (h, w), cfg
    )


class Share:
    """
    The RobotCar data share: log in once, then stream stereo chunks.

    Only the frames needed are kept from each chunk.

    Args:
        username: RobotCar account user name.
        password: RobotCar account password.

    Attributes:
        session: The logged-in HTTP session.

    Raises:
        SystemExit: The login page has no CSRF token, or the login failed.
    """

    def __init__(self, username: str, password: str):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "tcloc (research; https://github.com/mkaspr/tcloc)"
        )

        login = SHARE + "/accounts/login/"
        r = self.session.get(login, timeout=60)
        r.raise_for_status()

        found = re.search(r"name='csrfmiddlewaretoken' value='([^']+)'", r.text)
        token = self.session.cookies.get("csrftoken") or (found and found[1])

        if not token:
            raise SystemExit("RobotCar login page changed: no CSRF token")

        r = self.session.post(
            login,
            data={
                "csrfmiddlewaretoken": token,
                "username": username,
                "password": password,
                "next": "/",
            },
            headers={"Referer": login},
            timeout=60,
        )

        if (
            "sessionid" not in self.session.cookies
            or "/accounts/login" in r.url
        ):
            raise SystemExit(
                "RobotCar login failed: check ROBOTCAR_USERNAME and "
                "ROBOTCAR_PASSWORD"
            )

    def extract(
        self, traversal: str, side: str, chunk: int, wanted: set[int], out: Path
    ) -> int:
        """
        Stream ``<traversal>_stereo_<side>_<chunk>.tar``, saving the wanted
        frames.

        The tar is streamed and never stored. If the share's limit on
        concurrent downloads is reached, waits a minute and retries (up to 30
        times).

        Args:
            traversal: Traversal name, e.g. ``2014-12-09-13-21-02``.
            side: ``left`` or ``right``.
            chunk: Chunk number (from 1).
            wanted: Timestamps of the frames to keep.
            out: Folder to save them in as ``<timestamp>.png`` (raw Bayer
                images), created if missing.

        Returns:
            The number of frames saved.

        Raises:
            RuntimeError: The share answered with a web page instead of the
                file.
            requests.HTTPError: The download failed.
        """
        name = f"{traversal}_stereo_{side}_{chunk:02d}.tar"
        url = f"{SHARE}/download/?filename=datasets/{traversal}/{name}"

        for _ in range(30):
            r = self.session.get(url, stream=True, timeout=120)

            if "text/html" not in r.headers.get("Content-Type", ""):
                break

            page = r.text

            if "certain number of files" not in page:
                raise RuntimeError(
                    f"{name}: the share answered with a web page "
                    "instead of the file"
                )

            time.sleep(60)

        r.raise_for_status()
        r.raw.decode_content = True
        out.mkdir(parents=True, exist_ok=True)

        saved = 0

        with tarfile.open(fileobj=r.raw, mode="r|") as tar:
            for member in tar:
                stem = Path(member.name).stem

                if (
                    member.isfile()
                    and member.name.endswith(".png")
                    and stem.isdigit()
                    and int(stem) in wanted
                ):
                    data = tar.extractfile(member)
                    if data is not None:
                        (out / f"{stem}.png").write_bytes(data.read())
                        saved += 1

        return saved
