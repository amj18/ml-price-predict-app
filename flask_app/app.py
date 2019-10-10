from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import yaml

app = Flask(__name__)

# Configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_details = request.form
        recipe_name =  form_details['recipe_name']
        recipe_info = form_details['recipe_info']
        recipe_ingredients = form_details['recipe_ing']
        cur = mysql.connection.cursor() # Allows execution of queries to MySQL
        cur.execute("INSERT INTO recipes(recipe_name, recipe_info, "
                    "recipe_ingredients) VALUES("
                    "%s, %s, %s)", (recipe_name, recipe_info, recipe_ingredients))
        mysql.connection.commit()
        cur.close()
        return "Success: 200"
    return render_template('index.html')


@app.route('/recipes')
def recipes():
    cur = mysql.connection.cursor()
    results = cur.execute("SELECT * FROM recipes")
    if results > 0:
        form_details = cur.fetchall()
        return render_template('recipes.html', form_details=form_details)


if __name__ == '__main__':
    app.run(debug=True)