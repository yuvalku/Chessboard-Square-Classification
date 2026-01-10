import numpy as np
import chess

# Class encoding per spec
# White: P,R,N,B,Q,K -> 0..5
# Black: p,r,n,b,q,k -> 6..11
# Empty -> 12
PIECE_TO_ID = {
    chess.PAWN: 0,
    chess.ROOK: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Inverse mapping of your encoding
ID_TO_PIECE = {
    0: (chess.PAWN,   chess.WHITE),
    1: (chess.ROOK,   chess.WHITE),
    2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE),
    4: (chess.QUEEN,  chess.WHITE),
    5: (chess.KING,   chess.WHITE),
    6: (chess.PAWN,   chess.BLACK),
    7: (chess.ROOK,   chess.BLACK),
    8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK),
    10:(chess.QUEEN,  chess.BLACK),
    11:(chess.KING,   chess.BLACK),
}

def fen_to_grid_ids(fen: str) -> np.ndarray:
    """
    Convert a FEN string into an (8,8) grid of class IDs.
    Convention:
      grid[0,0] corresponds to the top-left square in the rendered board image.
    """
    board = chess.Board(fen)
    grid = np.full((8, 8), 12, dtype=np.int64)  # default empty

    # python-chess squares: a1=0 ... h8=63
    # We want grid[0,0] = a8 (top-left), grid[7,7] = h1 (bottom-right)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_id = PIECE_TO_ID[piece.piece_type]
        if piece.color == chess.BLACK:
            piece_id += 6  # shift black pieces to 6..11

        file = chess.square_file(square)  # a=0..h=7
        rank = chess.square_rank(square)  # 1=0..8=7 (rank1 -> 0)
        row = 7 - rank                    # rank8 -> row0
        col = file

        grid[row, col] = piece_id

    return grid

def grid_ids_to_fen(grid: np.ndarray, unknown_as_empty: bool = True) -> str:
    """
    Convert an (8,8) grid of class IDs into a FEN string.

    unknown_as_empty:
        True  -> treat 13 as empty (safe, standard FEN)
        False -> raise error if unknown exists
    """
    assert grid.shape == (8, 8)

    board = chess.Board.empty()

    for row in range(8):
        for col in range(8):
            cls = int(grid[row, col])

            if cls == 12:
                continue  # empty

            if cls == 13:
                if unknown_as_empty:
                    continue
                else:
                    raise ValueError("Unknown square present in grid")

            piece_type, color = ID_TO_PIECE[cls]

            rank = 7 - row          # row0 -> rank8
            file = col              # col0 -> file a
            square = chess.square(file, rank)

            board.set_piece_at(square, chess.Piece(piece_type, color))

    return board.fen()