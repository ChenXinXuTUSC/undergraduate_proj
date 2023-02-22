from plyfile import PlyData, PlyElement

if __name__ == "__main__":
    pdata = PlyData.read("./test/dummy_ply.ply")
    lines = [e for e in pdata.elements]
    print(lines.index("element"))