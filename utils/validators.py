def validate_inputs(c, A, b):
    """
    Valide les entrées
    """
    try:
        if len(c) != 2:
            return False
        if len(A) != len(b):
            return False
        for val in c:
            float(val)
        for row in A:
            for val in row:
                float(val)
        for val in b:
            float(val)
        return True
    except:
        return False

