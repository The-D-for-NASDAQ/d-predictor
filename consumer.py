import datetime
import main
import pika


def callback(ch, method, props, body):
    main.main(datetime.datetime.fromisoformat(body.decode('utf-8')))


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='predictor')
# TODO: set proper auto_ack to do not miss unprocessed events
channel.basic_consume(queue='predictor', on_message_callback=callback, auto_ack=True)
print('Ready to predict ✍(◔◡◔)')
channel.start_consuming()
