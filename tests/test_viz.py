"""tapeline viz on a synthetic scene (CPU): a box room as a point cloud, three cameras, a visibility map."""
import numpy as np
import pytest
from PIL import Image

from tapeline import viz
from tapeline.cli import _parser, main


def _room(n=20000, seed=0):
    """4 x 3 x 2.5 m box: floor, ceiling and walls, coloured by height."""
    g = np.random.default_rng(seed)
    P = g.uniform([0, 0, 0], [4, 3, 2.5], (n, 3))
    k = g.integers(0, 6, n)
    P[k == 0, 2] = 0.0; P[k == 1, 2] = 2.5
    P[k == 2, 0] = 0.0; P[k == 3, 0] = 4.0
    P[k == 4, 1] = 0.0; P[k == 5, 1] = 3.0
    rgb = viz.ramp(P[:, 2] / 2.5)
    return P.astype(np.float32), rgb


def _scene(tmp_path, wh=(160, 90)):
    xyz, rgb = _room()
    eyes = [(1.0, 1.0, 1.5), (2.0, 1.5, 1.5), (3.0, 2.0, 1.5)]
    targets = [(4.0, 1.0, 1.2), (2.0, 3.0, 1.2), (0.0, 2.0, 1.2)]
    Rt = [viz.look_at(e, t) for e, t in zip(eyes, targets)]
    K = np.array([[100.0, 0, wh[0] / 2], [0, 100.0, wh[1] / 2], [0, 0, 1]])
    names = [f"{i:06d}.png" for i in range(3)]
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for (R, t), nm in zip(Rt, names):
        Image.fromarray(viz.splat(xyz, rgb, R, t, K, wh, point=1, size_m=0.03)[0]).save(img_dir / nm)
    gx, gy = np.meshgrid(np.arange(0.02, 4, 0.04), np.arange(0.02, 3, 0.04))
    free = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 30.0)], 1)[: gx.size // 2]   # half the room observed
    return dict(xyz=xyz, rgb=rgb, R=np.stack([r for r, _ in Rt]), t=np.stack([t for _, t in Rt]), K=K, wh=wh,
                names=names, images=img_dir, free=free.astype(np.float32), cell=0.04, band=(0.5, 2.0))


def test_splat_projects_a_point_to_its_pixel_and_keeps_the_nearest():
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    R, t = np.eye(3), np.zeros(3)
    xyz = np.array([[0.2, -0.1, 2.0], [0.4, -0.2, 4.0]])   # same ray, the first is nearer
    rgb = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    img, cov = viz.splat(xyz, rgb, R, t, K, (100, 80), point=1)
    assert tuple(img[35, 60]) == (255, 0, 0)            # u = 100*0.1 + 50, v = 100*(-0.05) + 40
    assert cov == pytest.approx(1 / 8000)
    img, _ = viz.splat(xyz[::-1], rgb[::-1], R, t, K, (100, 80), point=1)
    assert tuple(img[35, 60]) == (255, 0, 0)            # order of points does not matter


def test_look_at_points_the_optical_axis_at_the_target():
    R, t = viz.look_at((1.0, 2.0, 3.0), (4.0, 2.0, 3.0))
    assert np.allclose(R @ np.array([4.0, 2.0, 3.0]) + t, [0, 0, 3.0])
    assert np.allclose(viz.centres(R[None], t[None])[0], [1.0, 2.0, 3.0])


def test_render_writes_all_outputs_with_expected_sizes(tmp_path):
    import av
    sc = _scene(tmp_path)
    res = viz.render(sc, tmp_path / "out", fps=5, height=90, gif_fps=4, gif_seconds=1, gif_width=420,
                     hero_size=(320, 180), log=lambda *_: None)
    assert set(res) == {"triptych", "gif", "hero", "topview", "visibility"}
    with av.open(res["triptych"]) as c:
        frames = [f.to_image() for f in c.decode(video=0)]
    assert len(frames) == 3
    assert frames[0].size == (160 + 160 + 90, 90)      # frame | render | square top view
    g = Image.open(res["gif"])
    assert g.n_frames == 3 and g.size[1] == 420 // (2 * 16 / 9 + 1) // 2 * 2
    assert Image.open(res["hero"]).size == (320, 180)
    assert max(Image.open(res["topview"]).size) == 1000
    vis = np.asarray(Image.open(res["visibility"]).convert("RGB"))
    assert vis.shape[0] > 100 and vis.shape[1] > 100 and vis.std() > 10


def test_triptych_middle_panel_matches_the_frame_for_the_true_pose(tmp_path):
    sc = _scene(tmp_path)
    top = viz.TopView(sc, (90, 90))
    im = np.asarray(viz.triptych_frame(sc, 1, 90, top, labels=False)).astype(float)
    frame, render = im[:, :160], im[:, 160:320]
    other = np.asarray(Image.open(sc["images"] / sc["names"][0]).convert("RGB")).astype(float)
    assert np.abs(frame - render).mean() < 0.25 * np.abs(other - render).mean()


def test_cli_parses_viz_with_a_positional_work_directory():
    a = _parser().parse_args(["viz", "runs/room", "--only", "hero", "gif", "--gif-seconds", "12"])
    assert (a.command, a.target, a.only, a.gif_seconds) == ("viz", "runs/room", ["hero", "gif"], 12.0)
    assert _parser().parse_args(["run", "--video", "v.mp4"]).target is None


def test_cli_rejects_a_positional_work_directory_outside_viz():
    with pytest.raises(SystemExit):
        main(["plan", "runs/room", "--video", "v.mp4"])
