import tensorflow as tf


def xywh_to_xyxy(bb):
    """
        Convert bounding boxes from xywh to xyxy

        Parameters
        ----------
        bb:  boxes tensor, shape=(batch, x, y, w, h),
        Returns
        -------
        new_bb: tensor, shape=(batch, xmin, ymin, xmax, ymax)
        """
    x, y, w, h = tf.unstack(bb, axis=-1)
    xmin = x
    ymin = y
    xmax = x + h
    ymax = y + w
    return tf.stack(xmin, ymin, xmax, ymax, axis=-1)

def xyxy_to_xywh(bb):
    """
        Convert bounding boxes from xyxy to xywh

        Parameters
        ----------
        bb:  boxes tensor, shape=(batch, xmin,ymin,xmax,ymax),
        Returns
        -------
        new_bb: tensor, shape=(batch, xmin,ymin,w,h)
        """
    xmin, ymin, xmax, ymax = tf.unstack(bb, axis=-1)
    h = xmax - xmin
    w = ymax - ymin
    return tf.stack(xmin, ymin, w, h , axis=-1)
