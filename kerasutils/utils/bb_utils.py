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
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


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
    return tf.stack([xmin, ymin, w, h], axis=-1)


def yolo_xywh_to_xyxy(bb):
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
    xmin, ymin = x - w / 2, y - h / 2
    xmax, ymax = x + w / 2, y + h / 2
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


def yolo_xyxy_to_xywh(bb):
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
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    h = tf.abs(xmax - xmin)
    w = tf.abs(ymax - ymin)
    return tf.stack([x, y, w, h], axis=-1)

# x = xyxy_to_xywh(tf.zeros((1, 1, 4)))
# print(x)
