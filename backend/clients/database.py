"""Database client using Prisma."""

from clients.prisma import Prisma

# Global Prisma client instance
db = Prisma()


async def connect_db() -> None:
    """Connect to the database."""
    if not db.is_connected():
        await db.connect()


async def disconnect_db() -> None:
    """Disconnect from the database."""
    if db.is_connected():
        await db.disconnect()


def get_db() -> Prisma:
    """Get the database client instance."""
    return db
