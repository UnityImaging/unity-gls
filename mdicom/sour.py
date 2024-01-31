def get_2d_area_from_sour(SequenceOfUltrasoundRegions = None):

    best_idx = None
    best_area = None

    if SequenceOfUltrasoundRegions is not None:
        for idx, x in enumerate(SequenceOfUltrasoundRegions):
            RegionLocationMinX0 = x.get('RegionLocationMinX0', None)
            RegionLocationMinY0 = x.get('RegionLocationMinY0', None)
            RegionLocationMaxX1 = x.get('RegionLocationMaxX1', None)
            RegionLocationMaxY1 = x.get('RegionLocationMaxY1', None)
            try:
                area = (RegionLocationMaxY1 - RegionLocationMinY0) * (RegionLocationMaxX1 - RegionLocationMinX0)
                if best_area is None or area > best_area:
                    # Use .value so if nothing generates exception
                    TempPhysicalUnitsXDirection = int(x['PhysicalUnitsXDirection'].value)
                    TempPhysicalUnitsYDirection = int(x['PhysicalUnitsYDirection'].value)
                    if TempPhysicalUnitsXDirection == 3 and TempPhysicalUnitsYDirection == 3:
                        best_area = area
                        best_idx = idx
            except Exception as e:
                pass

    if best_idx is None:
        return None

    out = {y.name.replace(" ", ""): y.value for y in SequenceOfUltrasoundRegions[best_idx]}

    return out


def get_spectral_area_from_sour(SequenceOfUltrasoundRegions = None):

    best_idx = None
    best_area = None

    if SequenceOfUltrasoundRegions is not None:
        for idx, x in enumerate(SequenceOfUltrasoundRegions):
            RegionLocationMinX0 = x.get('RegionLocationMinX0', None)
            RegionLocationMinY0 = x.get('RegionLocationMinY0', None)
            RegionLocationMaxX1 = x.get('RegionLocationMaxX1', None)
            RegionLocationMaxY1 = x.get('RegionLocationMaxY1', None)
            try:
                area = (RegionLocationMaxY1 - RegionLocationMinY0) * (RegionLocationMaxX1 - RegionLocationMinX0)
                if best_area is None or area > best_area:
                    # Use .value so if nothing generates exception
                    TempPhysicalUnitsXDirection = int(x['PhysicalUnitsXDirection'].value)
                    TempPhysicalUnitsYDirection = int(x['PhysicalUnitsYDirection'].value)
                    if TempPhysicalUnitsXDirection == 4 and TempPhysicalUnitsYDirection == 7:
                        best_area = area
                        best_idx = idx
            except Exception as e:
                pass

    if best_idx is None:
        return None

    out = {y.name.replace(" ", ""): y.value for y in SequenceOfUltrasoundRegions[best_idx]}

    return out


def get_mmode_area_from_sour(SequenceOfUltrasoundRegions = None):

    best_idx = None
    best_area = None

    if SequenceOfUltrasoundRegions is not None:
        for idx, x in enumerate(SequenceOfUltrasoundRegions):
            RegionLocationMinX0 = x.get('RegionLocationMinX0', None)
            RegionLocationMinY0 = x.get('RegionLocationMinY0', None)
            RegionLocationMaxX1 = x.get('RegionLocationMaxX1', None)
            RegionLocationMaxY1 = x.get('RegionLocationMaxY1', None)
            try:
                area = (RegionLocationMaxY1 - RegionLocationMinY0) * (RegionLocationMaxX1 - RegionLocationMinX0)
                if best_area is None or area > best_area:
                    # Use .value so if nothing generates exception
                    TempPhysicalUnitsXDirection = int(x['PhysicalUnitsXDirection'].value)
                    TempPhysicalUnitsYDirection = int(x['PhysicalUnitsYDirection'].value)
                    if TempPhysicalUnitsXDirection == 4 and TempPhysicalUnitsYDirection == 3:
                        best_area = area
                        best_idx = idx
            except Exception as e:
                pass

    if best_idx is None:
        return None

    out = {y.name.replace(" ", ""): y.value for y in SequenceOfUltrasoundRegions[best_idx]}

    return out


def convert_sour_to_json(SequenceOfUltrasoundRegions = None):
    out = []
    if SequenceOfUltrasoundRegions is not None:
        for x in SequenceOfUltrasoundRegions:
            out.append({y.name.replace(" ", ""): y.value for y in x})

    return out
