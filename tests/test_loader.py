"""Uji loader dua format, filter VoIP, dan rincian EDA."""
import copy
import math

import pandas as pd
import pytest

from qos.config import load_config
from qos.data.eda import (
    corrupt_counts,
    describe_by_group,
    describe_features,
    packet_medians,
    voip_breakdown,
    voip_by_group,
)
from qos.data.loader import (
    OUTPUT_COLUMNS,
    detect_format,
    detect_sep,
    filter_voip,
    label_mask,
    in_port_ranges,
    load_csv,
    load_flows,
    normalize_protocol,
    voip_port_masks,
)

CIC_CSV = (
    " Source Port, Destination Port, Protocol, Flow Duration, Total Fwd Packets,"
    " Total Backward Packets, Flow IAT Mean, Flow IAT Std, Label\n"
    "5060,5060,17,120540,150,148,810.5,812.45,BENIGN\n"
    "443,51000,6,450120,220,175,2050.0,3450.80,BENIGN\n"
    "12000,53,17,89400,110,109,500.0,Infinity,BENIGN\n"
)
BCCC_CSV = (
    "flow_id;src_port;dst_port;protocol;duration;fwd_packets_count;bwd_packets_count;packet_IAT_mean;packet_IAT_std;label\n"
    "a;5060;5060;UDP;0.12054;150;148;0.0008105;0.00081245;Benign\n"
    "b;443;51000;TCP;0.45012;220;175;0.00205;0.0034508;Benign\n"
    "c;12000;53;UDP;10.996.118.068.695.000;110;109;0.0005;7,36E-02;Benign\n"
)


DARKNET_CSV = (
    "Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Total Fwd Packet,"
    "Total Bwd packets,Flow IAT Mean,Flow IAT Std,Label,Label\n"
    "a,1.1.1.1,5060,2.2.2.2,5060,17,t,120540,150,148,810.5,812.45,VPN,VOIP\n"
    "b,1.1.1.1,443,2.2.2.2,51000,6,t,450120,220,175,2050.0,3450.80,Tor,Voip \n"
    "c,1.1.1.1,53,2.2.2.2,40000,17,t,89400,110,109,500.0,abc,NonVPN,Browsing\n"
    "d,1.1.1.1,53,2.2.2.2,40000,17,t,89400,110,109,100.0,5.0,Non-Tor,P2P\n"
)


@pytest.fixture
def cfg():
    """Config default; filter VoIP diset ke mode port agar tes lama tetap berlaku."""
    c = copy.deepcopy(load_config(use_user=False))
    c["data"]["voip_filter"]["mode"] = "port"
    return c


@pytest.fixture
def darknet_path(tmp_path):
    p = tmp_path / "darknet.csv"
    p.write_text(DARKNET_CSV, encoding="utf-8")
    return p


@pytest.fixture
def cic_path(tmp_path):
    p = tmp_path / "cic.csv"
    p.write_text(CIC_CSV, encoding="utf-8")
    return p


@pytest.fixture
def bccc_path(tmp_path):
    p = tmp_path / "bccc.csv"
    p.write_text(BCCC_CSV, encoding="utf-8")
    return p


def _flows(rows):
    """Bangun DataFrame internal dari (protocol, src_port, dst_port)."""
    df = pd.DataFrame(rows, columns=["protocol", "src_port", "dst_port"])
    for col in ("duration_us", "iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count"):
        df[col] = 1.0
    return df


def test_detect_sep():
    assert detect_sep("a;b;c,d") == ";"
    assert detect_sep(" Source Port, Destination Port, Protocol") == ","
    assert detect_sep("a\tb\tc") == "\t"


def test_detect_format(cfg):
    cmap = cfg["data"]["column_map"]
    assert detect_format(CIC_CSV.splitlines()[0].split(","), cmap) == "cic"
    assert detect_format(BCCC_CSV.splitlines()[0].split(";"), cmap) == "bccc"
    with pytest.raises(ValueError, match="tidak dikenali"):
        detect_format(["Flow Duration", "Protocol"], cmap)


def test_normalize_protocol():
    out = normalize_protocol(pd.Series(["17", " udp ", "TCP", "6", "???", None]))
    assert out.tolist()[:4] == [17.0, 17.0, 6.0, 6.0]
    assert out.isna().tolist()[4:] == [True, True]


def test_load_cic_satuan_us(cfg, cic_path):
    df = load_csv(cic_path, cfg["data"])
    assert tuple(df.columns) == OUTPUT_COLUMNS
    assert df.loc[0, "duration_us"] == 120540
    assert df.loc[0, "iat_std_us"] == pytest.approx(812.45)
    assert df.loc[0, "iat_mean_us"] == pytest.approx(810.5)
    assert df["protocol"].tolist() == [17, 6, 17]
    assert df.loc[0, "fwd_packets_count"] == 150 and df.loc[0, "bwd_packets_count"] == 148
    # "Infinity" tetap inf; pembersihan inf → NaN dilakukan di tahap cleaning (M2)
    assert math.isinf(df.loc[2, "iat_std_us"])


def test_load_bccc_detik_ke_us(cfg, bccc_path):
    df = load_csv(bccc_path, cfg["data"])
    assert tuple(df.columns) == OUTPUT_COLUMNS
    assert df.loc[0, "duration_us"] == pytest.approx(120540)
    assert df.loc[0, "iat_std_us"] == pytest.approx(812.45)
    assert df.loc[0, "iat_mean_us"] == pytest.approx(810.5)
    assert df["protocol"].tolist() == [17, 6, 17]
    assert df["src_port"].tolist() == [5060, 443, 12000]


def test_nilai_rusak_menjadi_nan(cfg, bccc_path):
    df = load_csv(bccc_path, cfg["data"])
    assert pd.isna(df.loc[2, "duration_us"])
    assert pd.isna(df.loc[2, "iat_std_us"])
    assert df.loc[2, "fwd_packets_count"] == 110


def test_time_unit_dan_sep_eksplisit(cfg, bccc_path):
    cfg["data"]["time_unit"] = "ms"
    cfg["data"]["csv_sep"] = ";"
    df = load_csv(bccc_path, cfg["data"])
    assert df.loc[0, "duration_us"] == pytest.approx(120.54)
    cfg["data"]["time_unit"] = "jam"
    with pytest.raises(ValueError, match="time_unit"):
        load_csv(bccc_path, cfg["data"])


def test_load_flows_folder(cfg, tmp_path, cic_path, bccc_path):
    df = load_flows(tmp_path, cfg["data"])
    assert len(df) == 6
    assert len(load_flows(cic_path, cfg["data"])) == 3
    empty = tmp_path / "kosong"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_flows(empty, cfg["data"])


def test_in_port_ranges_batas_inklusif():
    ports = pd.Series([9999, 10000, 20000, 20001, 32767, 32768])
    assert in_port_ranges(ports, [[10000, 20000]]).tolist() == [False, True, True, False, False, False]
    assert in_port_ranges(ports, [[10000, 20000], [16384, 32767]]).tolist() == [
        False, True, True, True, True, False,
    ]


def test_voip_port_masks(cfg):
    df = _flows([(17, 5060, 40000), (17, 40000, 5061), (17, 15000, 80), (17, 80, 443)])
    sip, rtp = voip_port_masks(df, cfg["data"]["voip_filter"])
    assert sip.tolist() == [True, True, False, False]
    assert rtp.tolist() == [False, False, True, False]


def test_filter_voip(cfg):
    voip = cfg["data"]["voip_filter"]
    voip["min_flows_warning"] = 0
    df = _flows([
        (17, 5060, 5060),    # SIP
        (17, 40000, 12000),  # RTP
        (6, 5060, 5060),     # TCP → dibuang
        (17, 53, 40000),     # UDP di luar port → dibuang
        (17, 9999, 32768),   # tepat di luar batas → dibuang
    ])
    out = filter_voip(df, voip)
    assert out[["src_port", "dst_port"]].values.tolist() == [[5060, 5060], [40000, 12000]]
    assert out.index.tolist() == [0, 1]


def test_filter_voip_peringatan_dan_fallback(cfg):
    voip = cfg["data"]["voip_filter"]
    df = _flows([(17, 5060, 5060), (17, 53, 40000), (6, 80, 443)])
    with pytest.warns(UserWarning, match="min_flows_warning"):
        assert len(filter_voip(df, voip)) == 1
    voip["fallback_udp_all"] = True
    voip["min_flows_warning"] = 0
    assert len(filter_voip(df, voip)) == 2


def test_voip_breakdown(cfg):
    voip = cfg["data"]["voip_filter"]
    suspect = cfg["eda"]["suspect_ports"]
    df = _flows([
        (17, 5060, 5060),    # SIP
        (17, 5061, 53),      # SIP + port curiga
        (17, 12000, 53),     # hanya RTP, lawan 53
        (17, 53, 15000),     # hanya RTP, lawan 53
        (17, 123, 30000),    # hanya RTP, lawan 123
        (17, 12000, 13000),  # hanya RTP, kedua port di rentang
        (17, 11000, 443),    # hanya RTP, lawan 443
        (17, 53, 40000),     # UDP bukan kandidat
        (6, 12000, 53),      # TCP
    ])
    b = voip_breakdown(df, voip, suspect)
    assert (b["total"], b["udp"], b["voip"]) == (9, 8, 7)
    assert (b["sip"], b["sip_suspect"]) == (2, 1)
    assert (b["rtp_only"], b["rtp_only_suspect"], b["rtp_only_both_in_range"]) == (5, 3, 1)
    assert b["suspect_by_port"][53] == 2 and b["suspect_by_port"][123] == 1
    assert b["suspect_by_port"][1900] == 0
    assert b["top_counter_ports"][0] == (53, 2)
    assert dict(b["top_counter_ports"]) == {53: 2, 123: 1, 443: 1}
    assert b["voip"] == len(filter_voip(df, {**voip, "min_flows_warning": 0}))


def test_describe_features():
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, None, -1.0], "b": [1.0, 1.0, 1.0, 1.0, 1.0]})
    d = describe_features(df, ["a", "b"], [0.5])
    assert d.loc["a", "nan"] == 1 and d.loc["a", "nol"] == 1 and d.loc["a", "negatif"] == 1
    assert d.loc["a", "count"] == 4 and d.loc["a", "50%"] == pytest.approx(0.5)
    assert d.loc["b", "mean"] == 1.0


def test_default_filter_mode_label():
    assert load_config(use_user=False)["data"]["voip_filter"]["mode"] == "label"


def test_detect_format_darknet(cfg):
    cmap = cfg["data"]["column_map"]
    # header darknet memuat "Label" dua kali; pandas menamai duplikat "Label.1"
    assert detect_format(DARKNET_CSV.splitlines()[0].split(",")[:-1] + ["Label.1"], cmap) == "darknet"


def test_load_darknet_label_dan_satuan(cfg, darknet_path):
    df = load_csv(darknet_path, cfg["data"])
    assert tuple(df.columns) == (*OUTPUT_COLUMNS, "traffic_type", "app_category")
    assert df.loc[0, "duration_us"] == 120540  # CICFlowMeter sudah µs
    assert df["fwd_packets_count"].tolist() == [150, 220, 110, 110]
    assert df["src_port"].tolist() == [5060, 443, 53, 53]
    assert df["traffic_type"].tolist() == ["VPN", "Tor", "NonVPN", "Non-Tor"]
    assert df["app_category"].tolist() == ["VOIP", "Voip", "Browsing", "P2P"]  # spasi di-strip
    assert pd.isna(df.loc[2, "iat_std_us"])  # nilai rusak → NaN


def test_filter_voip_mode_label(cfg, darknet_path):
    voip = {**cfg["data"]["voip_filter"], "mode": "label", "min_flows_warning": 0}
    df = load_csv(darknet_path, cfg["data"])
    assert label_mask(df, voip).tolist() == [True, True, False, False]
    out = filter_voip(df, voip)
    assert out["traffic_type"].tolist() == ["VPN", "Tor"]  # protokol TCP tidak mempengaruhi
    voip["label_value"] = " voip "
    assert len(filter_voip(df, voip)) == 2


def test_filter_voip_mode_label_tanpa_kolom_label(cfg):
    voip = {**cfg["data"]["voip_filter"], "mode": "label"}
    with pytest.raises(ValueError, match="kategori aplikasi"):
        filter_voip(_flows([(17, 5060, 5060)]), voip)


def test_filter_voip_mode_tak_dikenal(cfg):
    with pytest.raises(ValueError, match="mode"):
        filter_voip(_flows([(17, 5060, 5060)]), {**cfg["data"]["voip_filter"], "mode": "xyz"})


def _grouped():
    df = pd.DataFrame({
        "traffic_type": ["VPN", "VPN", "Tor", "Non-Tor"],
        "app_category": ["VOIP", "VOIP", "VOIP", "P2P"],
        "protocol": [17, 17, 6, 17],
        "src_port": [1, 2, 3, 4], "dst_port": [1, 2, 3, 4],
        "duration_us": [10.0, 20.0, float("inf"), 5.0],
        "iat_std_us": [0.0, 1.0, 2.0, float("nan")],
        "fwd_packets_count": [4, 6, 8, 1],
        "bwd_packets_count": [0, 10, 8, 1],
    })
    return df


def test_voip_by_group():
    df = _grouped()
    out = voip_by_group(df, df[df["app_category"] == "VOIP"])
    assert out.loc["VPN"].tolist() == [2, 2, 100.0]
    assert out.loc["Non-Tor"].tolist() == [1, 0, 0.0]


def test_corrupt_counts():
    c = corrupt_counts(_grouped())
    assert c.loc["duration_us", "inf"] == 1
    assert c.loc["iat_std_us", "nan"] == 1 and c.loc["iat_std_us", "nol"] == 1
    assert c.loc["bwd_packets_count", "nol"] == 1 and c["negatif"].sum() == 0


def test_describe_by_group_inf_jadi_nan():
    d = describe_by_group(_grouped(), ["duration_us"], [0.5])
    assert set(d) == {"Non-Tor", "Tor", "VPN"}
    assert d["Tor"].loc["duration_us", "count"] == 0 and d["Tor"].loc["duration_us", "nan"] == 1
    assert d["VPN"].loc["duration_us", "50%"] == pytest.approx(15.0)


def test_packet_medians():
    m = packet_medians(_grouped())
    assert m.loc["VPN", "median_fwd"] == 5 and m.loc["VPN", "median_bwd"] == 5
    assert m.loc["VPN", "bwd_nol"] == 1 and m.loc["VPN", "bwd_lebih_besar"] == 1
    assert m.loc["(semua)", "n"] == 4
